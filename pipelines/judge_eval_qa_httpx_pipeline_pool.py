import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

from commons.utils_msg import msg_debug, msg_error, msg_info


RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
JUDGE_LABELS = {"correct", "partially_correct", "incorrect", "unjudgeable"}


@dataclass
class JudgeJob:
    item_id: int
    step: int
    payload: Dict[str, Any]
    previous_outputs: Dict[str, Any] = field(default_factory=dict)


def entry_cache_key(entry_id: str, chunk_index: Any) -> str:
    if chunk_index is None:
        return entry_id
    return f"{entry_id}\t{chunk_index}"


def is_entry_processed(entry_id: str, chunk_index: Any, processed_keys: set[str]) -> bool:
    key = entry_cache_key(entry_id, chunk_index)
    return key in processed_keys or entry_id in processed_keys


class JudgeEvalQAPipeline:
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.inference_config = dict(settings.get("infer_config", {}))

        if settings.get("openrouter", False):
            api_key = settings.get("openrouter_api_key", "dummy")
            server_url = settings.get("openrouter_server_url", "https://openrouter.ai/api/v1")
            model_name = settings.get("openrouter_model_name", None)
            self.runtime_label = "openrouter"
        else:
            api_key = "dummy"
            server_url = settings.get("SERVER_URL", "http://localhost:8000/v1")
            model_name = settings.get("MODEL_NAME", None)
            self.runtime_label = "local"

        self.inference_config.update(
            {
                "API_KEY": api_key,
                "SERVER_URL": str(server_url).rstrip("/"),
                "MODEL_NAME": model_name,
            }
        )

        self.output_dir = Path(settings.get("output_path", "./test_output/eval_judge")).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.id_key = str(settings.get("id_key", "id"))
        self.question_key = str(settings.get("question_key", "question"))
        self.reference_answer_key = str(settings.get("reference_answer_key", "answer"))
        self.candidate_answer_key = str(settings.get("candidate_answer_key", "answer"))
        self.batch_size = int(settings.get("batch_size", 1))
        self.max_in_flight = max(1, int(settings.get("max_in_flight", self.batch_size)))
        self.pipeline_batch_size = int(
            settings.get("pipeline_batch_size", max(self.batch_size, self.max_in_flight * 4))
        )
        self.max_retries = int(settings.get("max_retries", 8))
        self.wait_seconds = float(settings.get("wait_seconds", 0.25))
        self.retry_jitter_seconds = float(settings.get("retry_jitter_seconds", self.wait_seconds))
        self.retry_max_delay = float(settings.get("retry_max_delay", 30.0))
        self.thinking_enabled_by_step = self._load_thinking_enabled_by_step(
            settings.get("thinking_enabled_by_step", {})
        )
        self.prompts = self._load_prompts(settings.get("prompts", []))

        self.request_semaphore = asyncio.Semaphore(self.max_in_flight)
        self._in_flight_lock = asyncio.Lock()
        self.current_in_flight = 0
        self.max_observed_in_flight = 0

        timeout_total = float(settings.get("read_timeout", self.inference_config.get("timeout", 600.0)))
        connect_timeout = float(settings.get("connect_timeout", 5.0))
        pool_timeout = float(settings.get("pool_timeout", 30.0))
        max_connections = int(settings.get("max_connections", max(16, self.max_in_flight * 2)))
        max_keepalive = int(settings.get("max_keepalive_connections", max(8, self.max_in_flight)))

        self.client = httpx.AsyncClient(
            base_url=self.inference_config.get("SERVER_URL"),
            headers={
                "Authorization": f"Bearer {self.inference_config.get('API_KEY', 'dummy')}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=timeout_total,
                write=30.0,
                pool=pool_timeout,
            ),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive,
                keepalive_expiry=float(settings.get("keepalive_expiry", 120.0)),
            ),
            http2=bool(settings.get("http2", False)),
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def judge_dataset(self, eval_qa_path: Path, candidate_answer_path: Path) -> Dict[str, int]:
        eval_entries = self.load_entries(eval_qa_path)
        answer_entries = self.load_entries(candidate_answer_path)
        answer_index = self.build_answer_index(answer_entries)

        all_path = self.output_dir / "all.jsonl"
        failure_path = self.output_dir / "all.failures.jsonl"
        cache_path = self.output_dir / "cache_processed_ids.txt"
        stats_path = self.output_dir / "stats.json"
        all_path.touch(exist_ok=True)
        cache_path.touch(exist_ok=True)

        processed_keys = self.load_processed_cache_keys(cache_path)
        candidates = self.prepare_candidates(eval_entries, answer_index, processed_keys)
        missing_answers = len(eval_entries) - len(candidates)

        print(
            msg_info(
                f"Loaded eval_qa={len(eval_entries)}, answers={len(answer_entries)}, "
                f"pending={len(candidates)}, skipped_or_missing={missing_answers}"
            )
        )

        stats = await self.judge_qa_async(
            candidates,
            all_path=all_path,
            failure_path=failure_path,
            cache_path=cache_path,
        )
        stats["loaded_eval_qa"] = len(eval_entries)
        stats["loaded_candidate_answers"] = len(answer_entries)
        stats["pending_records"] = len(candidates)
        stats["skipped_or_missing_answers"] = missing_answers
        stats["failed_records"] = self._count_jsonl_records(failure_path)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        return stats

    def load_entries(self, source_path: Path) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for file_path in self._collect_source_files(source_path):
            entries.extend(self._load_file(file_path))
        return entries

    def build_answer_index(self, answer_entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        answer_index: Dict[str, Dict[str, Any]] = {}
        for row_index, entry in enumerate(answer_entries):
            entry_id = str(entry.get(self.id_key) or f"{entry.get('source_file', '')}:{row_index}")
            cache_key = entry_cache_key(entry_id, entry.get("chunk_index"))
            answer_index[cache_key] = entry
            answer_index.setdefault(entry_id, entry)
        return answer_index

    def prepare_candidates(
        self,
        eval_entries: List[Dict[str, Any]],
        answer_index: Dict[str, Dict[str, Any]],
        processed_keys: set[str],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for row_index, eval_entry in enumerate(eval_entries):
            entry_id = str(eval_entry.get(self.id_key) or f"{eval_entry.get('source_file', '')}:{row_index}")
            chunk_index = eval_entry.get("chunk_index")
            cache_key = entry_cache_key(entry_id, chunk_index)
            if is_entry_processed(entry_id, chunk_index, processed_keys):
                continue

            if chunk_index is None:
                answer_entry = answer_index.get(cache_key) or answer_index.get(entry_id)
            else:
                answer_entry = answer_index.get(cache_key)
            if not answer_entry:
                continue

            question = self._as_text(eval_entry.get(self.question_key))
            reference_answer = self._as_text(eval_entry.get(self.reference_answer_key))
            candidate_answer = self._as_text(answer_entry.get(self.candidate_answer_key))
            if not question or not reference_answer or not candidate_answer:
                continue

            candidates.append(
                {
                    "id": entry_id,
                    "chunk_index": chunk_index,
                    "question": question,
                    "reference_answer": reference_answer,
                    "candidate_answer": candidate_answer,
                    "cache_key": cache_key,
                }
            )
        return candidates

    async def judge_qa_async(
        self,
        candidates: List[Dict[str, Any]],
        all_path: Path,
        failure_path: Path,
        cache_path: Path,
    ) -> Dict[str, int]:
        self._validate_prompts()
        if not candidates:
            return {
                "saved_records": 0,
                "failed_items": 0,
                "correct_records": 0,
                "partially_correct_records": 0,
                "incorrect_records": 0,
                "unjudgeable_records": 0,
                "max_observed_in_flight": self.max_observed_in_flight,
            }

        queue: asyncio.Queue[Optional[int]] = asyncio.Queue()
        output_lock = asyncio.Lock()
        saved_count = 0
        failed_count = 0
        label_counts = {label: 0 for label in JUDGE_LABELS}
        start_time = time.monotonic()

        for item_id in range(len(candidates)):
            await queue.put(item_id)

        worker_count = max(1, min(self.max_in_flight, len(candidates)))
        print(
            msg_info(
                "Judging evaluation QA with pipeline-pool async httpx pipeline: "
                f"items={len(candidates)}, workers={worker_count}, "
                f"max_in_flight={self.max_in_flight}, input_window_hint={self.pipeline_batch_size}"
            )
        )

        def build_initial_job(item_id: int) -> JudgeJob:
            return JudgeJob(item_id=item_id, step=1, payload=dict(candidates[item_id]))

        async def save_success(result: Dict[str, Any]) -> None:
            nonlocal saved_count
            async with output_lock:
                self._append_jsonl_record(all_path, result)
                self._append_processed_key(cache_path, str(result["cache_key"]))
                saved_count += 1
                label_counts[result.get("judge_label", "unjudgeable")] += 1

        async def save_failure(candidate: Dict[str, Any], step: int, error: str, outputs: Dict[str, Any]) -> None:
            nonlocal failed_count
            async with output_lock:
                failed_count += 1
                self._append_jsonl_record(
                    failure_path,
                    {
                        "id": candidate.get("id", ""),
                        "chunk_index": candidate.get("chunk_index"),
                        "question": candidate.get("question", ""),
                        "failed": True,
                        "failed_step": step,
                        "error": error,
                        "previous_outputs": outputs,
                        "judge_model": self.inference_config.get("MODEL_NAME", ""),
                    },
                )

        async def worker(worker_id: int) -> None:
            while True:
                item_id = await queue.get()
                if item_id is None:
                    queue.task_done()
                    return

                job = build_initial_job(item_id)
                item_start = time.monotonic()
                try:
                    job = await self._run_one_step(job)
                    await save_success(self._final_result(job.item_id, job.payload, job.previous_outputs))
                    print(
                        msg_info(
                            f"Judge job complete worker_id={worker_id} item_id={job.item_id} "
                            f"elapsed={time.monotonic() - item_start:.2f}s in_flight={self.current_in_flight}"
                        )
                    )
                except Exception as exc:
                    print(
                        msg_error(
                            f"Judge job failed worker_id={worker_id} item_id={job.item_id} "
                            f"step={job.step} elapsed={time.monotonic() - item_start:.2f}s error={exc}"
                        )
                    )
                    await save_failure(job.payload, job.step, str(exc), job.previous_outputs)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker(worker_id)) for worker_id in range(worker_count)]
        await queue.join()
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

        total_elapsed = time.monotonic() - start_time
        print(
            msg_info(
                f"Judge pipeline-pool finished items={len(candidates)} saved={saved_count} "
                f"failed={failed_count} elapsed={total_elapsed:.2f}s "
                f"max_observed_in_flight={self.max_observed_in_flight}"
            )
        )
        return {
            "saved_records": saved_count,
            "failed_items": failed_count,
            "correct_records": label_counts["correct"],
            "partially_correct_records": label_counts["partially_correct"],
            "incorrect_records": label_counts["incorrect"],
            "unjudgeable_records": label_counts["unjudgeable"],
            "max_observed_in_flight": self.max_observed_in_flight,
        }

    async def _run_one_step(self, job: JudgeJob) -> JudgeJob:
        prompt_template = self.prompts["judge_prompt"]
        raw_text = await self._infer_text_async(
            prompt_template.format(
                question=job.payload.get("question", ""),
                reference_answer=job.payload.get("reference_answer", ""),
                candidate_answer=job.payload.get("candidate_answer", ""),
            ),
            step=job.step,
        )
        outputs = self._parse_judge_output(raw_text)
        return JudgeJob(
            item_id=job.item_id,
            step=job.step + 1,
            payload=job.payload,
            previous_outputs=outputs,
        )

    def _final_result(self, item_id: int, payload: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "item_id": item_id,
            "id": payload.get("id", ""),
            "chunk_index": payload.get("chunk_index"),
            "question": payload.get("question", ""),
            "reference_answer": payload.get("reference_answer", ""),
            "candidate_answer": payload.get("candidate_answer", ""),
            "judge_score": outputs["judge_score"],
            "judge_label": outputs["judge_label"],
            "judge_reason": outputs["judge_reason"],
            "judge_model": self.inference_config.get("MODEL_NAME", ""),
            "cache_key": payload.get("cache_key", ""),
        }

    def _parse_judge_output(self, raw_text: str) -> Dict[str, Any]:
        score_text = self._extract_tag(raw_text, "judge_score")
        label = self._extract_tag(raw_text, "judge_label").strip().lower()
        reason = self._extract_tag(raw_text, "judge_reason")
        try:
            score = int(score_text)
        except ValueError as exc:
            raise ValueError("Judge returned invalid judge_score.") from exc
        if score < 1 or score > 5:
            raise ValueError("Judge returned judge_score outside 1-5.")
        if label not in JUDGE_LABELS:
            raise ValueError("Judge returned invalid judge_label.")
        if not reason:
            raise ValueError("Judge returned empty judge_reason.")
        return {
            "judge_score": score,
            "judge_label": label,
            "judge_reason": reason,
        }

    def _validate_prompts(self) -> None:
        if not self.prompts.get("judge_prompt"):
            raise ValueError("Missing prompt settings: judge_prompt")

    def _chat_payload(self, prompt: str, step: Optional[int] = None) -> Dict[str, Any]:
        payload = {
            "model": self.inference_config.get("MODEL_NAME"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(self.inference_config.get("max_tokens", 2048)),
            "temperature": self.inference_config.get("temperature", 0),
            "top_p": self.inference_config.get("top_p", 1.0),
            "stream": False,
        }
        if step in self.thinking_enabled_by_step:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.thinking_enabled_by_step[step],
            }
        return payload

    async def _post_chat_completion(self, payload: Dict[str, Any]) -> httpx.Response:
        return await self.client.post("/chat/completions", json=payload)

    async def _infer_text_async(self, prompt: str, step: Optional[int] = None) -> str:
        payload = self._chat_payload(prompt, step=step)
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                async with self.request_semaphore:
                    await self._increment_in_flight()
                    try:
                        response = await self._post_chat_completion(payload)
                    finally:
                        await self._decrement_in_flight()

                if response.status_code in RETRYABLE_STATUS_CODES:
                    raise httpx.HTTPStatusError(
                        f"Retryable status code: {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                response.raise_for_status()
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content")
                normalized = self._normalize_content(content)
                if normalized:
                    return normalized
                raise ValueError("Model returned blank text.")

            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code not in RETRYABLE_STATUS_CODES:
                    body = exc.response.text[:500] if exc.response is not None else ""
                    print(msg_error(f"Non-retryable HTTP error: status={status_code}, body={body}"))
                    return ""
            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.PoolTimeout,
                httpx.ReadError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                last_exc = exc
            except Exception as exc:
                print(msg_error(f"Inference failed with unexpected error: {exc}"))
                return ""

            if attempt < self.max_retries - 1:
                response = last_exc.response if isinstance(last_exc, httpx.HTTPStatusError) else None
                delay = self._retry_delay(attempt, response)
                reason = self._retry_reason(last_exc) if last_exc is not None else "unknown"
                print(
                    msg_debug(
                        f"Retrying judge inference ({attempt + 1}/{self.max_retries}) "
                        f"after {delay:.2f}s. reason={reason}"
                    )
                )
                await asyncio.sleep(delay)

        print(msg_error(f"Inference failed after {self.max_retries} attempts. last_error={last_exc}"))
        return ""

    async def _increment_in_flight(self) -> int:
        async with self._in_flight_lock:
            self.current_in_flight += 1
            self.max_observed_in_flight = max(self.max_observed_in_flight, self.current_in_flight)
            return self.current_in_flight

    async def _decrement_in_flight(self) -> int:
        async with self._in_flight_lock:
            self.current_in_flight = max(0, self.current_in_flight - 1)
            return self.current_in_flight

    def _retry_delay(self, attempt: int, response: Optional[httpx.Response] = None) -> float:
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return min(float(retry_after), self.retry_max_delay)
                except ValueError:
                    pass
        base_delay = self.wait_seconds * (2**attempt)
        jitter = random.random() * self.retry_jitter_seconds
        return min(base_delay + jitter, self.retry_max_delay)

    def _retry_reason(self, exc: Exception) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            body = exc.response.text[:200] if exc.response is not None else ""
            return f"{type(exc).__name__}: status={status_code}, body={body}"
        return f"{type(exc).__name__}: {exc}"

    def _load_prompts(self, prompts_settings: List[Dict[str, str]]) -> Dict[str, str]:
        prompts_dict: Dict[str, str] = {}
        for prompt_path_dict in prompts_settings:
            key, prompt_path = list(prompt_path_dict.items())[0]
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompts_dict[key] = f.read()
        return prompts_dict

    def _load_thinking_enabled_by_step(self, raw_settings: Dict[str, Any]) -> Dict[int, Optional[bool]]:
        step_aliases = {
            1: ("1", "step1", "step_1", "judge", "judge_prompt"),
        }
        if not isinstance(raw_settings, dict):
            return {}

        parsed: Dict[int, Optional[bool]] = {}
        for step, aliases in step_aliases.items():
            for alias in aliases:
                if alias in raw_settings:
                    parsed[step] = self._parse_optional_bool(raw_settings[alias])
                    break
        return parsed

    def _parse_optional_bool(self, value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
        return bool(value)

    def _normalize_content(self, content: object) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    value = part.get("text")
                    if value:
                        text_parts.append(str(value))
                else:
                    text_parts.append(str(part))
            return "".join(text_parts).strip()
        return str(content).strip()

    def _extract_tag(self, text: Optional[str], tag: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        if start_tag in text and end_tag in text:
            text = text.split(start_tag)[-1]
            text = text.split(end_tag)[0]
        elif start_tag in text:
            text = text.split(start_tag)[-1]
        elif end_tag in text:
            text = text.split(end_tag)[0]
        else:
            return ""
        return text.strip()

    def _collect_source_files(self, source_path: Path) -> List[Path]:
        if source_path.is_file():
            return [source_path]
        if source_path.is_dir():
            return sorted(
                p
                for p in source_path.rglob("*")
                if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
            )
        raise FileNotFoundError(f"Source path not found: {source_path}")

    def _load_file(self, file_path: Path) -> List[Dict[str, Any]]:
        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            return list(self._load_jsonl(file_path))
        if suffix == ".json":
            return self._load_json(file_path)
        print(msg_debug(f"Skipping unsupported file: {file_path}"))
        return []

    def _load_jsonl(self, file_path: Path) -> Iterable[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print(msg_debug(f"Skipping invalid JSONL row {line_no} in {file_path.name}."))
                    continue
                if isinstance(obj, dict):
                    obj.setdefault("source_file", str(file_path))
                    yield obj

    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            entries = [entry for entry in raw if isinstance(entry, dict)]
        elif isinstance(raw, dict):
            entries = [raw]
        else:
            entries = []
        for entry in entries:
            entry.setdefault("source_file", str(file_path))
        return entries

    def _count_jsonl_records(self, file_path: Path) -> int:
        if not file_path.exists():
            return 0
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def load_processed_cache_keys(self, cache_path: Path) -> set[str]:
        if not cache_path.exists():
            return set()
        with open(cache_path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())

    def _append_processed_key(self, cache_path: Path, cache_key: str) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(f"{cache_key}\n")

    def _append_jsonl_record(self, save_path: Path, record: Dict[str, Any]) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    def _as_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()
