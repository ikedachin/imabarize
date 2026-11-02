"""Generate answers with a served model and grade them with a separate LLM."""

import asyncio
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import httpx

from pipelines.judge_eval_qa import JudgeEvalQAPipeline, RETRYABLE_STATUS_CODES


class EvaluationError(Exception):
    pass


DEFAULT_STYLES = {
    "imabari": {"label": "今治弁", "valid_channels": ["analysis_imabari", "final_imabari", "commentary"]},
    "standard": {"label": "標準語", "valid_channels": ["analysis", "final", "commentary"]},
}
DEFAULT_EFFORTS = ["low", "medium", "high"]
STYLE_LABELS = {"matches", "partially_matches", "does_not_match", "unjudgeable"}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def load_dataset(path: Path, fields: dict[str, str]) -> list[dict[str, Any]]:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl"})
    else:
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    if not files:
        raise EvaluationError(f"No JSON or JSONL files found: {path}")
    records, identities = [], set()
    for file in files:
        if file.suffix.lower() == ".jsonl":
            with file.open(encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
        elif file.suffix.lower() == ".json":
            with file.open(encoding="utf-8") as stream:
                data = json.load(stream)
            rows = data if isinstance(data, list) else [data]
        else:
            raise EvaluationError(f"Unsupported dataset file: {file}")
        for number, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise EvaluationError(f"Expected an object at {file}, record {number}")
            raw_id = row.get(fields["id"])
            question = row.get(fields["question"])
            answer = row.get(fields["answer"])
            if raw_id is None or not str(raw_id).strip() or not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
                raise EvaluationError(f"Missing ID, question or reference answer at {file}, record {number}")
            entry_id = str(raw_id)
            chunk = row.get("chunk_index")
            identity = _canonical([entry_id, chunk])
            if identity in identities:
                raise EvaluationError(f"Duplicate id and chunk_index: {entry_id!r}, {chunk!r}")
            identities.add(identity)
            records.append({"id": entry_id, "chunk_index": chunk, "question": question.strip(), "reference_answer": answer.strip(), "key": _digest([entry_id, chunk])})
    if not records:
        raise EvaluationError("Evaluation dataset is empty")
    return records


def extract_final_answer(content: Any, finish_reason: Any) -> tuple[str, str | None]:
    """Accept parsed content or a complete Harmony/thinking-formatted answer."""
    if finish_reason == "length":
        raise EvaluationError("Answer was truncated (finish_reason=length)")
    if not isinstance(content, str) or not content.strip():
        raise EvaluationError("Model returned no final answer in message.content")
    text = content.strip()
    observed_channel = None
    if "<|channel|>" in text or "<|start|>" in text:
        matches = re.findall(r"<\|channel\|>(final(?:_imabari)?)<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>|<\|start\|>|$)", text, flags=re.DOTALL)
        if not matches:
            raise EvaluationError("Harmony response has no final channel")
        observed_channel, text = matches[-1]
        text = text.strip()
    else:
        # Some servers strip Harmony special tokens but leave role/channel text.
        # Only treat it as a delimiter when the response begins in an analysis
        # channel (or directly with an assistant final header).
        analysis_prefix = re.match(r"^(?:assistant\s+)?analysis(?:_imabari)?(?=\s|$)", text)
        direct_final = re.match(r"^assistant\s+(final(?:_imabari)?)(?=\s|$)", text)
        if analysis_prefix:
            finals = list(re.finditer(r"(?:^|\s)assistant\s+(final(?:_imabari)?)(?=\s|$)", text))
            if not finals:
                raise EvaluationError("Analysis response has no final channel")
            final = finals[-1]
            observed_channel, text = final.group(1), text[final.end():].strip()
        elif direct_final:
            observed_channel, text = direct_final.group(1), text[direct_final.end():].strip()
    if "<think>" in text or "</think>" in text:
        if text.count("<think>") != text.count("</think>") or "</think>" not in text:
            raise EvaluationError("Unfinished thinking in model response")
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = text.replace("<|return|>", "").replace("<|end|>", "").strip()
    if not text:
        raise EvaluationError("Model returned an empty final answer")
    return text, observed_channel


def final_answer(content: Any, finish_reason: Any) -> str:
    return extract_final_answer(content, finish_reason)[0]


def build_cases(records: list[dict[str, Any]], settings: dict[str, Any]) -> list[dict[str, Any]]:
    styles = settings.get("styles", DEFAULT_STYLES)
    efforts = settings.get("reasoning_efforts", DEFAULT_EFFORTS)
    if not isinstance(styles, dict) or not styles:
        raise EvaluationError("styles must be a non-empty mapping")
    if not isinstance(efforts, list) or not efforts or any(e not in DEFAULT_EFFORTS for e in efforts) or len(set(efforts)) != len(efforts):
        raise EvaluationError("reasoning_efforts must be unique values from low, medium, high")
    cases = []
    for style, config in styles.items():
        if not isinstance(style, str) or not style or not isinstance(config, dict):
            raise EvaluationError("Each style must be a named mapping")
        label, channels = config.get("label"), config.get("valid_channels")
        if not isinstance(label, str) or not label.strip() or not isinstance(channels, list) or not channels or any(not isinstance(c, str) or not c for c in channels):
            raise EvaluationError(f"Style {style!r} requires a label and valid_channels")
        for effort in efforts:
            for row in records:
                cases.append({**row, "key": _digest([row["id"], row["chunk_index"], style, effort]), "style": style, "style_label": label, "reasoning_effort": effort, "valid_channels": channels})
    return cases


class ChatClient:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model = str(config["model"])
        self.attempts = int(config.get("max_attempts", 4))
        self.wait = float(config.get("retry_wait_seconds", 0.5))
        self.limit = int(config.get("max_in_flight", 4))
        if self.attempts < 1 or self.limit < 1 or self.wait < 0:
            raise EvaluationError("max_attempts and max_in_flight must be positive; retry_wait_seconds must be non-negative")
        key_env = config.get("api_key_env")
        key = os.environ.get(key_env) if key_env else "dummy"
        if key_env and not key:
            raise EvaluationError(f"API key environment variable is missing: {key_env}")
        self.semaphore = asyncio.Semaphore(self.limit)
        self.client = httpx.AsyncClient(base_url=str(config["base_url"]).rstrip("/") + "/", headers={"Authorization": f"Bearer {key}"}, timeout=httpx.Timeout(float(config.get("timeout", 600)), connect=float(config.get("connect_timeout", 10))), limits=httpx.Limits(max_connections=self.limit, max_keepalive_connections=self.limit))

    async def close(self) -> None:
        await self.client.aclose()

    async def complete(self, prompt: str, *, extract_final: bool, template_kwargs: dict[str, Any] | None = None) -> tuple[str, str | None]:
        payload: dict[str, Any] = {**self.config.get("generation", {}), "model": self.model, "messages": [{"role": "user", "content": prompt}], "stream": False}
        if template_kwargs is not None:
            payload["chat_template_kwargs"] = {**payload.get("chat_template_kwargs", {}), **template_kwargs}
        error: Exception | None = None
        for attempt in range(self.attempts):
            retry_after = None
            try:
                async with self.semaphore:
                    response = await self.client.post("chat/completions", json=payload)
                if response.status_code in RETRYABLE_STATUS_CODES:
                    retry_after = response.headers.get("retry-after")
                    response.raise_for_status()
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                if extract_final:
                    return extract_final_answer(choice["message"].get("content"), choice.get("finish_reason"))
                if choice.get("finish_reason") == "length":
                    raise EvaluationError("Judge response was truncated (finish_reason=length)")
                content = choice["message"].get("content")
                if not isinstance(content, str) or not content.strip():
                    raise EvaluationError("Judge returned empty message.content")
                return content.strip(), None
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in RETRYABLE_STATUS_CODES:
                    raise EvaluationError(f"HTTP {exc.response.status_code} from {self.model}") from exc
                error = exc
            except (httpx.RequestError, json.JSONDecodeError, KeyError, IndexError, TypeError, EvaluationError) as exc:
                error = exc
            if attempt + 1 < self.attempts:
                try:
                    delay = min(float(retry_after), 30) if retry_after is not None else min(self.wait * 2**attempt + random.random() * self.wait, 30)
                except ValueError:
                    delay = min(self.wait * 2**attempt, 30)
                await asyncio.sleep(max(0, delay))
        raise EvaluationError(f"{self.model} failed after {self.attempts} attempts: {type(error).__name__}: {error}")


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result = {}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                key = row["key"]
                if key in result:
                    raise EvaluationError(f"Duplicate saved key at {path}:{number}")
                result[key] = row
    return result


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


async def evaluate(settings: dict[str, Any], dataset_path: Path, stage: str = "both") -> dict[str, Any]:
    if stage not in {"generate", "judge", "both"}:
        raise EvaluationError(f"Unknown stage: {stage}")
    fields = {"id": "id", "question": "question", "answer": "answer", **settings.get("fields", {})}
    dataset = load_dataset(dataset_path, fields)
    records = build_cases(dataset, settings)
    target_config = settings["target"]
    output = Path(settings.get("output_path", "./test_output/llm_jp_evaluation")).expanduser().resolve()
    workers = int(settings.get("workers", 8))
    if workers < 1:
        raise EvaluationError("workers must be positive")
    manifest_path = output / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    # Judge settings are fixed when judging starts, not when answers are generated.
    judge_config, judge_prompt, style_prompt = None, None, None
    if stage != "generate" or (previous and "fingerprint" in previous):
        judge_config = settings["judge"]
        prompt_path = Path(settings.get("judge_prompt", "./prompts/judge_eval_qa/judge_answer.md")).expanduser().resolve()
        judge_prompt = prompt_path.read_text(encoding="utf-8")
        style_prompt_path = Path(settings.get("style_judge_prompt", "./prompts/judge_eval_qa/judge_llm_jp_style.md")).expanduser().resolve()
        style_prompt = style_prompt_path.read_text(encoding="utf-8")
        if "{style_label}" not in style_prompt:
            raise EvaluationError("Style judge prompt missing {style_label}")
        for field in ("question", "reference_answer", "candidate_answer"):
            if "{" + field + "}" not in judge_prompt:
                raise EvaluationError(f"Judge prompt missing {{{field}}}")
    generation_fingerprint = _digest({"version": 3, "dataset": records, "target": target_config})
    judge_fingerprint = _digest({"judge": judge_config, "judge_prompt": judge_prompt, "style_prompt": style_prompt}) if judge_config is not None else None
    if manifest_path.exists():
        if "fingerprint" in previous:
            old = {"version": 2, "dataset": records, "target": target_config, "judge": judge_config, "judge_prompt": judge_prompt, "style_prompt": style_prompt}
            if previous["fingerprint"] != _digest(old):
                raise EvaluationError("Dataset or evaluation settings changed; choose a new output_path")
        elif previous.get("generation_fingerprint") != generation_fingerprint:
            raise EvaluationError("Dataset or generation settings changed; choose a new output_path")
        elif judge_fingerprint and previous.get("judge_fingerprint") not in (None, judge_fingerprint):
            raise EvaluationError("Judge settings changed; choose a new output_path")
    else:
        if stage == "judge":
            raise EvaluationError("No generated answers found; run --stage generate first")
        if any((output / name).exists() for name in ("answers.jsonl", "all.jsonl", "failures.jsonl")):
            raise EvaluationError("Output files exist without manifest.json; choose a new output_path")
        output.mkdir(parents=True, exist_ok=True)
        previous = {"version": 3, "generation_fingerprint": generation_fingerprint, "dataset_path": str(dataset_path), "target_model": target_config["model"], "records": len(records)}
        manifest_path.write_text(json.dumps(previous, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    answers_path, results_path, failures_path = (output / name for name in ("answers.jsonl", "all.jsonl", "failures.jsonl"))
    answers, results = read_jsonl(answers_path), read_jsonl(results_path)
    keys = {row["key"] for row in records}
    if (answers.keys() | results.keys()) - keys:
        raise EvaluationError("Output contains records absent from the current dataset")
    if stage == "judge" and not answers:
        raise EvaluationError("No generated answers found; run --stage generate first")
    target = ChatClient(target_config) if stage != "judge" else None
    try:
        judge = ChatClient(judge_config) if stage != "generate" else None
    except BaseException:
        if target is not None:
            await target.close()
        raise
    try:
        if judge_fingerprint and "fingerprint" not in previous and "judge_fingerprint" not in previous:
            previous.update(judge_fingerprint=judge_fingerprint, judge_model=judge_config["model"])
            manifest_path.write_text(json.dumps(previous, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        await asyncio.gather(*(client.close() for client in (target, judge) if client is not None))
        raise
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=max(workers * 2, 1))
    lock = asyncio.Lock()
    counts = {"generation_failures": 0, "judge_failures": 0, "generated_this_run": 0, "judged_this_run": 0}

    async def worker(number: int) -> None:
        while (item := await queue.get()) is not None:
            phase = "generation"
            try:
                key = item["key"]
                if key in answers:
                    answer = answers[key]["answer"]
                    observed_channel = answers[key].get("observed_final_channel")
                else:
                    answer, observed_channel = await target.complete(item["question"], extract_final=True, template_kwargs={"valid_channels": item["valid_channels"], "reasoning_effort": item["reasoning_effort"]})
                    async with lock:
                        row = {"key": key, "id": item["id"], "chunk_index": item["chunk_index"], "question": item["question"], "answer": answer, "model": target.model, "style": item["style"], "reasoning_effort": item["reasoning_effort"], "valid_channels": item["valid_channels"], "observed_final_channel": observed_channel}
                        append_jsonl(answers_path, row)
                        answers[key] = row
                        counts["generated_this_run"] += 1
                if stage == "generate":
                    print(f"worker={number} id={item['id']} style={item['style']} effort={item['reasoning_effort']} generated", flush=True)
                    continue
                phase = "judge"
                prompt = judge_prompt.format(question=item["question"], reference_answer=item["reference_answer"], candidate_answer=answer)
                prompt += "\n\n" + style_prompt.format(style_label=item["style_label"])
                raw, _ = await judge.complete(prompt, extract_final=False)
                # Reuse the existing judge parser without creating another HTTP client.
                judged = parse_judgement(raw)
                result = {**item, "candidate_answer": answer, "observed_final_channel": observed_channel, **judged, "target_model": target_config["model"], "judge_model": judge.model}
                async with lock:
                    append_jsonl(results_path, result)
                    results[key] = result
                    counts["judged_this_run"] += 1
                print(f"worker={number} id={item['id']} style={item['style']} effort={item['reasoning_effort']} label={judged['judge_label']}", flush=True)
            except Exception as exc:
                async with lock:
                    counts[f"{phase}_failures"] += 1
                    append_jsonl(failures_path, {"key": item["key"], "id": item["id"], "chunk_index": item["chunk_index"], "style": item["style"], "reasoning_effort": item["reasoning_effort"], "stage": phase, "error": f"{type(exc).__name__}: {exc}"})
                print(f"worker={number} id={item['id']} stage={phase} failed: {exc}", flush=True)
            finally:
                queue.task_done()
        queue.task_done()

    async def produce() -> None:
        for item in records:
            key = item["key"]
            if ((stage == "generate" and key not in answers)
                or (stage == "judge" and key in answers and key not in results)
                or (stage == "both" and key not in results)):
                await queue.put(item)
        for _ in range(workers):
            await queue.put(None)

    tasks = [asyncio.create_task(worker(i)) for i in range(workers)]
    tasks.append(asyncio.create_task(produce()))
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*(client.close() for client in (target, judge) if client is not None))
    labels = {name: sum(row["judge_label"] == name for row in results.values()) for name in ("correct", "partially_correct", "incorrect", "unjudgeable")}
    groups = {}
    for item in records:
        group = groups.setdefault(item["style"], {}).setdefault(item["reasoning_effort"], {"total": 0, "generated": 0, "judged": 0, "labels": {label: 0 for label in labels}, "style_labels": {label: 0 for label in sorted(STYLE_LABELS)}, "observed_final_channels": {"final": 0, "final_imabari": 0, "unknown": 0}, "mean_score": None})
        group["total"] += 1
        group["generated"] += item["key"] in answers
        if item["key"] in answers:
            channel = answers[item["key"]].get("observed_final_channel") or "unknown"
            group["observed_final_channels"][channel] += 1
        if item["key"] in results:
            result = results[item["key"]]
            group["judged"] += 1
            group["labels"][result["judge_label"]] += 1
            group["style_labels"][result["style_judge_label"]] += 1
            group["mean_score"] = (group["mean_score"] or 0) + result["judge_score"]
    for efforts in groups.values():
        for group in efforts.values():
            if group["judged"]:
                group["mean_score"] /= group["judged"]
    stats = {"stage": stage, "input_questions": len(dataset), "total_records": len(records), "generated_records": len(answers), "judged_records": len(results), "remaining_generation_records": len(records) - len(answers), "remaining_records": len(records) - len(results), **counts, "labels": labels, "mean_score": sum(row["judge_score"] for row in results.values()) / len(results) if results else None, "by_style_and_effort": groups}
    (output / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def parse_judgement(raw: str) -> dict[str, Any]:
    # These methods do not depend on pipeline state.
    class Parser:
        _extract_tag = JudgeEvalQAPipeline._extract_tag
    parser = Parser()
    judged = JudgeEvalQAPipeline._parse_judge_output(parser, raw)
    label = parser._extract_tag(raw, "style_judge_label").lower()
    reason = parser._extract_tag(raw, "style_judge_reason")
    if label not in STYLE_LABELS or not reason:
        raise EvaluationError("Judge returned invalid style label or empty style reason")
    return {**judged, "style_judge_label": label, "style_judge_reason": reason}
