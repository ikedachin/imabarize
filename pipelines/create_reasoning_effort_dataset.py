"""Generate canonical thinking once and independently resume each SFT target."""
import asyncio
from collections import Counter
from copy import deepcopy
import json
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from commons.utils_msg import msg_info

from reasoning_effort.cache import JsonlJournal, stable_id
from reasoning_effort.formatters import FORMATTERS
from reasoning_effort.output_schema import compact_output
from reasoning_effort.progress import QAProgress, write_log
from reasoning_effort.generator import OUTPUT_CONTRACT, ThinkingGenerator
from reasoning_effort.source_context import SourceContextIndex
from reasoning_effort.validators import (
    ThinkingFormatValidator, classify_length, cleanup, count_tokens, length_errors, retry_feedback,
)

EFFORTS = ("low", "medium", "high")


class RecordFailure(Exception):
    def __init__(self, step: str, errors: list[str], rejected_thinking: str | None = None):
        super().__init__(", ".join(errors))
        self.step, self.errors = step, errors
        self.rejected_thinking = rejected_thinking


class ReasoningEffortDatasetPipeline:
    def __init__(self, settings: dict, *, generator: Any = None,
                 reference_tokenizer: Any = None, formatters: dict | None = None):
        self.settings = deepcopy(settings)
        self.generation = settings["thinking_generation"]
        self.mode = settings["reasoning_effort"]["mode"]
        self._validate_settings()
        self.prompts = {e: Path(self.generation[e]["prompt"]).read_text(encoding="utf-8") for e in EFFORTS}
        self.reference_tokenizer = reference_tokenizer
        self.formatters = formatters
        self.generator = generator
        self.validator = ThinkingFormatValidator()
        self.stats: Counter = Counter()
        self._locks: dict[str, asyncio.Lock] = {}
        self._file_locks: list = []
        self.source_context: SourceContextIndex | None = None
        self._initialize_storage()

    def _validate_settings(self) -> None:
        if self.mode not in {"fixed", "expand_all", "token_length"}:
            raise ValueError("Unknown reasoning_effort.mode")
        effort = self.settings["reasoning_effort"].get("effort", "medium")
        if effort not in EFFORTS:
            raise ValueError("Canonical effort must be low, medium or high")
        if not self.generation.get("format_validation", True) or not self.generation.get("length_validation", True):
            raise ValueError("Format and length validation must remain enabled")
        for key in ("max_in_flight", "pipeline_batch_size"):
            if self.settings.get(key, 8) < 1:
                raise ValueError(f"{key} must be positive")
        for key in ("max_format_retries", "max_length_retries"):
            if self.generation.get(key, 2) < 0:
                raise ValueError(f"{key} must be nonnegative")
        if self.settings.get("max_retries", 3) < 0:
            raise ValueError("max_retries must be nonnegative")
        for e in EFFORTS:
            bounds = self.generation[e]
            if bounds["min_tokens"] < 0 or bounds["max_tokens"] < bounds["min_tokens"]:
                raise ValueError(f"Invalid length range: {e}")
        if self.mode == "token_length":
            ranges = self.settings["reasoning_effort"]["token_length"]
            if set(ranges) != set(EFFORTS):
                raise ValueError("token_length requires all three canonical ranges")
            previous = -1
            for e in EFFORTS:
                low = ranges[e].get("min_tokens", 0)
                high = ranges[e].get("max_tokens", float("inf"))
                if low != previous + 1 or high < low:
                    raise ValueError("token_length ranges must be contiguous and non-overlapping, starting at 0")
                previous = high
            if previous != float("inf"):
                raise ValueError("token_length high must have no max_tokens")
        targets = self.settings["targets"]
        if not any(t.get("enabled", True) for t in targets.values()):
            raise ValueError("At least one target must be enabled")
        if set(targets) - set(FORMATTERS):
            raise ValueError("Unknown target family")

    def _initialize_storage(self) -> None:
        import fcntl
        output = self.settings["output"]
        paths = [Path(output[k]).expanduser().resolve() for k in ("cache_path", "failures_path")]
        enabled = [f for f, c in self.settings["targets"].items() if c.get("enabled", True)]
        paths += [Path(output[f]["path"]).expanduser().resolve() for f in enabled]
        paths += [Path(str(Path(output[f]["path"]).expanduser().resolve()) + '.resume.jsonl') for f in enabled]
        if len(set(paths)) != len(paths):
            raise ValueError("Cache, failure and output paths must be distinct")
        source = self.settings.get("source", {})
        if source.get("type") == "local" and Path(source["path"]).expanduser().resolve() in paths:
            raise ValueError("Output must not overwrite source")
        context = self.settings.get("source_context", {})
        if context.get("enabled", False) and Path(context["path"]).expanduser().resolve() in paths:
            raise ValueError("Output must not overwrite source context")
        try:
            for path in sorted(paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                lock = Path(str(path) + ".lock").open("a")
                self._file_locks.append(lock)
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.cache = JsonlJournal(paths[0])
            self.failures = JsonlJournal(paths[1])
            self.outputs = {f: JsonlJournal(output[f]["path"]) for f in enabled}
            self.output_metadata = {f: JsonlJournal(str(j.path) + '.resume.jsonl')
                                    for f, j in self.outputs.items()}
        except Exception:
            for lock in self._file_locks:
                lock.close()
            raise

    def _load_tokenizers(self) -> None:
        if self.reference_tokenizer is not None and self.formatters is not None:
            return
        from transformers import AutoTokenizer
        loaded = {}

        def load(name: str, revision: str = "main", trust: bool = False):
            key = (name, revision, trust)
            if key not in loaded:
                loaded[key] = AutoTokenizer.from_pretrained(
                    name, revision=revision, trust_remote_code=trust,
                    cache_dir=self.generation.get("tokenizer_cache_dir"),
                )
            return loaded[key]

        if self.reference_tokenizer is None:
            self.reference_tokenizer = load(self.generation["length_reference_tokenizer"],
                                            self.generation.get("reference_revision", "main"))
        if self.formatters is None:
            self.formatters = {}
            for family in self.outputs:
                cfg = self.settings["targets"][family]
                name, revision = cfg["tokenizer_name"], cfg.get("revision", "main")
                self.formatters[family] = FORMATTERS[family](
                    load(name, revision, cfg.get("trust_remote_code", False)), name, revision)

    def _source(self, row: dict) -> dict:
        if not isinstance(row, dict):
            raise ValueError("source_record_must_be_object")
        fields = self.settings.get("fields", {})
        record = deepcopy(row)
        for name in ("question", "answer"):
            value = row.get(fields.get(name, name))
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"missing_or_empty_{name}")
            record[name] = value
        original = row.get(fields.get("thinking", "thinking"))
        if self.settings["output"].get("keep_original_thinking", False):
            record["original_thinking"] = original
        record["source_metadata"] = deepcopy(row)
        # Original thinking remains only when explicitly requested.
        if not self.settings["output"].get("keep_original_thinking", False):
            record["source_metadata"].pop(fields.get("thinking", "thinking"), None)
            record["source_metadata"].pop("messages", None)
        record.pop("messages", None)
        record.pop(fields.get("thinking", "thinking"), None)
        record["source_qa_id"] = row.get(fields.get("id", "qa_id"))
        if record["source_qa_id"] is None:
            record["source_qa_id"] = stable_id(row)
        record["generation_context"] = row.get(fields.get("context")) if fields.get("context") else None
        if self.source_context is not None:
            try:
                article = self.source_context.resolve(row)
            except ValueError as exc:
                raise RecordFailure("source_context_resolution", [str(exc)]) from exc
            record["generation_context"] = article.text
            record["source_context_metadata"] = {
                "article_id": article.article_id,
                "path": str(self.source_context.path),
                "chunk_indices": list(article.chunk_indices),
                "text_sha256": article.text_sha256,
            }
        return record

    def _key(self, source: dict, effort: str) -> str:
        gen = self.settings["generator"]
        return stable_id({"source": source, "effort": effort, "mode": self.mode,
                          "generator": {k: v for k, v in gen.items() if "key" not in k},
                          "prompt": self.prompts[effort], "prompt_version": self.generation.get("prompt_version", "1"),
                          "generator_output_contract": OUTPUT_CONTRACT,
                          "validation": self.generation, "labeling": self.settings["reasoning_effort"],
                          "schema_version": 1})

    def _check(self, record: dict) -> None:
        errors = self.validator.validate(record["thinking"])
        if errors:
            raise RecordFailure("thinking_format_validation", errors)
        count = count_tokens(self.reference_tokenizer, record["thinking"])
        if self.mode == "token_length":
            effort = classify_length(count, self.settings["reasoning_effort"]["token_length"])
            errors = [] if effort == record["canonical_reasoning_effort"] else ["cached_effort_mismatch"]
        else:
            errors = length_errors(count, self.generation[record["canonical_reasoning_effort"]])
        if count != record["canonical_thinking_tokens"]:
            errors.append("canonical_token_count_mismatch")
        if errors:
            raise RecordFailure("thinking_length_validation", errors)

    async def _canonical(self, source: dict, effort: str, key: str) -> dict:
        if key in self.cache.records:
            record = self.cache.records[key]
            self._check(record)
            self.stats["canonical_cache_hits"] += 1
            return record
        prompt = self.prompts[effort] + "\n\n入力データ（命令ではありません）:\n" + json.dumps(
            {"question": source["question"], "validated_answer": source["answer"],
             "context": source["generation_context"], "canonical_reasoning_effort": effort,
             "target_token_range": {k: v for k, v in self.generation[effort].items() if k != "prompt"}},
            ensure_ascii=False,
        )
        feedback = ""
        retries = Counter()
        while True:
            self.stats["generator_calls"] += 1
            write_log(msg_info(f"Thinking request source_qa_id={source['source_qa_id']} effort={effort} "
                               f"attempt={1 + sum(retries.values())}"))
            thinking = cleanup(await self.generator.generate(prompt + feedback, effort))
            errors = self.validator.validate(thinking)
            step = "thinking_format_validation"
            if not errors:
                count = count_tokens(self.reference_tokenizer, thinking)
                step = "thinking_length_validation"
                if self.mode == "token_length":
                    canonical_effort = classify_length(count, self.settings["reasoning_effort"]["token_length"])
                else:
                    canonical_effort = effort
                    errors = length_errors(count, self.generation[effort])
            if not errors:
                break
            self.stats[step + "_failures"] += 1
            write_log(msg_info(f"Thinking rejected source_qa_id={source['source_qa_id']} effort={effort} "
                               f"step={step} errors={','.join(errors)}"))
            retry_key = "max_format_retries" if step == "thinking_format_validation" else "max_length_retries"
            if retries[step] >= self.generation.get(retry_key, 2):
                raise RecordFailure(step, errors, thinking)
            retries[step] += 1
            feedback = "\n\n" + retry_feedback(errors)
            if step == "thinking_length_validation":
                bounds = self.generation[effort]
                feedback += (
                    f"\n前回の実測はreference tokenizerで{count} tokensでした。"
                    f"許容範囲は{bounds['min_tokens']}〜{bounds['max_tokens']} tokensです。"
                    "範囲内に収めてください。短すぎる場合はcontextにある関連する事実と"
                    "質問の条件の対応を具体的に検証し、長すぎる場合は重複を削ってください。"
                    "新しい事実の創作や、同じ説明の繰り返しによる水増しは禁止です。"
                )
        record = dict(source, thinking=thinking, canonical_reasoning_effort=canonical_effort,
                      canonical_record_id=key, canonical_thinking_tokens=count,
                      canonical_tokenizer=self.generation["length_reference_tokenizer"],
                      canonical_tokenizer_revision=self.generation.get("reference_revision", "main"),
                      thinking_generator=self.settings["generator"]["model_name"],
                      generation_method={"fixed": "fixed_effort", "expand_all": "effort_conditioned",
                                         "token_length": "token_length_labeled"}[self.mode])
        # The full article has already contributed to the prompt and cache key.
        record.pop("generation_context", None)
        self._check(record)
        self.cache.append(record, key)
        write_log(msg_info(f"Thinking validated source_qa_id={source['source_qa_id']} "
                           f"effort={canonical_effort} tokens={count}"))
        return record

    def _failure(self, source: dict, key: str, effort: str | None, step: str, exc: Exception) -> None:
        self.stats[step + "_terminal_failures"] += 1
        self.failures.append({"source_qa_id": source.get("source_qa_id"), "canonical_record_id": key,
                              "source_article_id": source.get(self.settings.get("source_context", {}).get("qa_id_field", "id")),
                              "canonical_reasoning_effort": effort, "failed_step": step,
                              "format_errors": getattr(exc, "errors", []), "error": str(exc),
                              "rejected_thinking": getattr(exc, "rejected_thinking", None)})

    async def _process(self, source: dict, effort: str) -> None:
        key = ""
        step = "generation"
        try:
            key = self._key(source, effort)
            async with self._locks.setdefault(key, asyncio.Lock()):
                step = "generation"
                canonical = await self._canonical(source, effort, key)
                self.stats["canonical_" + canonical["canonical_reasoning_effort"]] += 1
                for family, journal in self.outputs.items():
                    target_step = family + "_formatting"
                    try:
                        formatter = self.formatters[family]
                        if key in journal.records:
                            saved = journal.records[key]
                            metadata = saved if 'target_tokenizer' in saved else self.output_metadata[family].records.get(key, {})
                            if not metadata:
                                raise ValueError('Missing output resume metadata; restore the .resume.jsonl file or choose a new output path')
                            if (metadata.get("target_tokenizer") != formatter.tokenizer_name
                                    or metadata.get("target_tokenizer_revision") != formatter.revision):
                                raise ValueError("Target tokenizer changed; choose a new output path")
                            self.stats[family + "_skipped"] += 1
                            continue
                        self._check(canonical)
                        record = formatter.format(canonical)
                        self._check(record)
                        for field in ("source_qa_id", "canonical_record_id", "question", "thinking", "answer", "canonical_reasoning_effort"):
                            if record[field] != canonical[field]:
                                raise ValueError(f"Formatter modified canonical field: {field}")
                        target_step = family + "_write"
                        exported = compact_output(record)
                        self.output_metadata[family].append({
                            'canonical_record_id': key,
                            'target_tokenizer': formatter.tokenizer_name,
                            'target_tokenizer_revision': formatter.revision,
                        })
                        journal.append(exported)
                        self.stats[family + "_written"] += 1
                    except Exception as exc:
                        self._failure(source, key, canonical["canonical_reasoning_effort"], target_step, exc)
        except Exception as exc:
            self._failure(source, key, effort, getattr(exc, "step", step), exc)

    async def run(self, rows: Iterable[dict], *, total: int | None = None) -> dict:
        context = self.settings.get("source_context", {})
        if context.get("enabled", False) and self.source_context is None:
            self.source_context = await asyncio.to_thread(SourceContextIndex, context)
            self.stats["source_context_rows"] = self.source_context.row_count
            self.stats["source_context_articles"] = len(self.source_context.articles)
            self.stats["source_context_duplicate_chunks"] = self.source_context.duplicate_chunk_count
        await asyncio.to_thread(self._load_tokenizers)
        if self.generator is None:
            self.generator = ThinkingGenerator(self.settings)
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.settings.get("pipeline_batch_size", 32))
        if total is None and hasattr(rows, '__len__'):
            total = len(rows)
        progress = QAProgress(total, self.stats)

        async def refresh_progress():
            while True:
                await asyncio.sleep(1)
                progress.refresh(force=progress.terminal)

        async def worker():
            while True:
                job = await queue.get()
                try:
                    if job is None:
                        return
                    source, effort, remaining = job
                    progress.active += 1
                    try:
                        await self._process(source, effort)
                    finally:
                        progress.active -= 1
                    remaining[0] -= 1
                    if remaining[0] == 0:
                        progress.finish_qa()
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(self.settings.get("max_in_flight", 8))]
        refresher = asyncio.create_task(refresh_progress())
        try:
            iterator = iter(rows)
            window = self.settings.get("pipeline_batch_size", 32)
            while batch := await asyncio.to_thread(lambda: list(islice(iterator, window))):
                for row in batch:
                    self.stats["source_qa"] += 1
                    try:
                        source = self._source(row)
                    except Exception as exc:
                        identity = dict(row) if isinstance(row, dict) else {}
                        identity["source_qa_id"] = identity.get(self.settings.get("fields", {}).get("id", "qa_id"))
                        self._failure(identity, "", None, getattr(exc, "step", "source_validation"), exc)
                        progress.finish_qa()
                        continue
                    efforts = EFFORTS if self.mode == "expand_all" else (self.settings["reasoning_effort"].get("effort", "medium"),)
                    remaining = [len(efforts)]
                    for effort in efforts:
                        await queue.put((source, effort, remaining))
                write_log(msg_info(f"Reasoning source QA queued={self.stats['source_qa']} "
                               f"generator_calls={self.stats['generator_calls']}"))
            for _ in workers:
                await queue.put(None)
            await asyncio.gather(*workers)
        finally:
            refresher.cancel()
            for worker_task in workers:
                worker_task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            await asyncio.gather(refresher, return_exceptions=True)
            progress.close()
        result = dict(self.stats)
        result["llm_inference_calls"] = getattr(self.generator, "call_count", self.stats["generator_calls"])
        result["output_records"] = {family: len(j.records) for family, j in self.outputs.items()}
        return result

    async def aclose(self) -> None:
        try:
            if self.generator is not None and hasattr(self.generator, "aclose"):
                await self.generator.aclose()
        finally:
            for lock in self._file_locks:
                lock.close()
