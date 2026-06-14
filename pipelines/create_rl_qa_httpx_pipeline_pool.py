import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx

from commons.utils_msg import msg_debug, msg_error, msg_info


RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
NUM_STEPS = 4
CHOICE_LABELS = ("A", "B", "C", "D")
DIFFICULTY_VALUES = {"borderline", "too_easy", "too_hard", "unknown"}
SUITABILITY_VALUES = {"accepted", "rejected"}


@dataclass
class RLQAPipelineJob:
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


class RLQAPipeline:
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

        self.output_dir = Path(settings.get("output_path", "./test_output/rl_qa/wiki")).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_key = str(settings.get("target_key", "text"))
        self.title_key = str(settings.get("title_key", "title"))
        self.id_key = str(settings.get("id_key", "id"))
        self.sample_size = max(1, int(settings.get("sample_size", 500)))
        self.seed = int(settings.get("seed", 42))
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

    async def build_dataset(self, source_path: Path) -> Dict[str, int]:
        entries = self.load_entries(source_path)
        all_path = self.output_dir / "all.jsonl"
        failure_path = self.output_dir / "all.failures.jsonl"
        cache_path = self.output_dir / "cache_processed_ids.txt"
        stats_path = self.output_dir / "stats.json"
        all_path.touch(exist_ok=True)
        cache_path.touch(exist_ok=True)

        processed_keys = self.load_processed_cache_keys(cache_path)
        candidates = self.prepare_candidates(entries, processed_keys)
        selected_candidates = self.sample_candidates(candidates)

        print(
            msg_info(
                f"Loaded entries={len(entries)}, pending={len(candidates)}, "
                f"selected={len(selected_candidates)}, sample_size={self.sample_size}"
            )
        )

        stats = await self.create_rl_qa_async(
            selected_candidates,
            all_path=all_path,
            failure_path=failure_path,
            cache_path=cache_path,
        )
        stats["loaded_entries"] = len(entries)
        stats["pending_entries"] = len(candidates)
        stats["selected_entries"] = len(selected_candidates)
        stats["failed_records"] = self._count_jsonl_records(failure_path)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        return stats

    def load_entries(self, source_path: Path) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for file_path in self._collect_source_files(source_path):
            entries.extend(self._load_file(file_path))
        return entries

    def prepare_candidates(
        self,
        entries: List[Dict[str, Any]],
        processed_keys: set[str],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for row_index, entry in enumerate(entries):
            text = entry.get(self.target_key)
            if not isinstance(text, str) or not text.strip():
                continue
            entry_id = str(entry.get(self.id_key) or f"{entry.get('source_file', '')}:{row_index}")
            chunk_index = entry.get("chunk_index")
            if is_entry_processed(entry_id, chunk_index, processed_keys):
                continue
            candidates.append(
                {
                    "id": entry_id,
                    "chunk_index": chunk_index,
                    "title": str(entry.get(self.title_key, "")),
                    "reference_text": self._normalize_text(text),
                    "cache_key": entry_cache_key(entry_id, chunk_index),
                }
            )
        return candidates

    def sample_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(candidates) <= self.sample_size:
            selected = list(candidates)
            random.Random(self.seed).shuffle(selected)
            return selected
        return random.Random(self.seed).sample(candidates, self.sample_size)

    async def create_rl_qa_async(
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
                "accepted_records": 0,
                "rejected_records": 0,
                "max_observed_in_flight": self.max_observed_in_flight,
            }

        queue: asyncio.Queue[Optional[int]] = asyncio.Queue()
        output_lock = asyncio.Lock()
        saved_count = 0
        failed_count = 0
        accepted_count = 0
        rejected_count = 0
        start_time = time.monotonic()

        for item_id in range(len(candidates)):
            await queue.put(item_id)

        worker_count = max(1, min(self.max_in_flight, len(candidates)))
        print(
            msg_info(
                "Generating RL QA with pipeline-pool async httpx pipeline: "
                f"items={len(candidates)}, workers={worker_count}, "
                f"max_in_flight={self.max_in_flight}, input_window_hint={self.pipeline_batch_size}"
            )
        )

        def build_initial_job(item_id: int) -> RLQAPipelineJob:
            payload = dict(candidates[item_id])
            text = str(payload.get("reference_text", ""))
            payload["random_token"] = "".join(random.sample(text, min(len(text), 10))) if text else ""
            return RLQAPipelineJob(item_id=item_id, step=1, payload=payload)

        async def save_success(result: Dict[str, Any]) -> None:
            nonlocal saved_count, accepted_count, rejected_count
            async with output_lock:
                self._append_jsonl_record(all_path, result)
                self._append_processed_key(cache_path, str(result["cache_key"]))
                saved_count += 1
                if result.get("rl_suitability") == "accepted":
                    accepted_count += 1
                else:
                    rejected_count += 1

        async def save_failure(candidate: Dict[str, Any], step: int, error: str, outputs: Dict[str, Any]) -> None:
            nonlocal failed_count
            async with output_lock:
                failed_count += 1
                self._append_jsonl_record(
                    failure_path,
                    {
                        "id": candidate.get("id", ""),
                        "chunk_index": candidate.get("chunk_index"),
                        "title": candidate.get("title", ""),
                        "failed": True,
                        "failed_step": step,
                        "error": error,
                        "previous_outputs": outputs,
                        "qa_generator": self.inference_config.get("MODEL_NAME", ""),
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
                    while job.step <= NUM_STEPS:
                        step_start = time.monotonic()
                        print(
                            msg_debug(
                                f"RL QA job start worker_id={worker_id} item_id={job.item_id} "
                                f"step={job.step} in_flight={self.current_in_flight}"
                            )
                        )
                        job = await self._run_one_step(job)
                        print(
                            msg_info(
                                f"RL QA job complete worker_id={worker_id} item_id={job.item_id} "
                                f"step={job.step - 1} elapsed={time.monotonic() - step_start:.2f}s "
                                f"in_flight={self.current_in_flight}"
                            )
                        )
                    await save_success(self._final_result(job.item_id, job.payload, job.previous_outputs))
                except Exception as exc:
                    print(
                        msg_error(
                            f"RL QA job failed worker_id={worker_id} item_id={job.item_id} "
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
                f"RL QA pipeline-pool finished items={len(candidates)} saved={saved_count} "
                f"accepted={accepted_count} rejected={rejected_count} failed={failed_count} "
                f"elapsed={total_elapsed:.2f}s max_observed_in_flight={self.max_observed_in_flight}"
            )
        )
        return {
            "saved_records": saved_count,
            "failed_items": failed_count,
            "accepted_records": accepted_count,
            "rejected_records": rejected_count,
            "max_observed_in_flight": self.max_observed_in_flight,
        }

    async def _run_one_step(self, job: RLQAPipelineJob) -> RLQAPipelineJob:
        payload = job.payload
        outputs = dict(job.previous_outputs)
        reference_text = str(payload.get("reference_text", ""))

        if job.step == 1:
            prompt_template = self.prompts["question_prompt"]
            raw_text = await self._infer_text_async(
                prompt_template.format(
                    text=reference_text,
                    title=payload.get("title", ""),
                    random_token=payload.get("random_token", ""),
                ),
                step=job.step,
            )
            question = self._extract_tag(raw_text, "question")
            if not question:
                raise ValueError("Step 1 returned empty question.")
            outputs["question"] = question

        elif job.step == 2:
            prompt_template = self.prompts["blind_answer_prompt"]
            raw_text = await self._infer_text_async(
                prompt_template.format(question=outputs.get("question", "")),
                step=job.step,
            )
            blind_answer = self._extract_tag(raw_text, "blind_answer")
            if not blind_answer:
                raise ValueError("Step 2 returned empty blind_answer.")
            outputs["blind_answer"] = blind_answer
            outputs["blind_confidence"] = self._extract_tag(raw_text, "blind_confidence")

        elif job.step == 3:
            prompt_template = self.prompts["grounded_answer_prompt"]
            raw_text = await self._infer_text_async(
                prompt_template.format(
                    text=reference_text,
                    title=payload.get("title", ""),
                    question=outputs.get("question", ""),
                ),
                step=job.step,
            )
            grounded_answer = self._extract_tag(raw_text, "grounded_answer")
            evidence = self._extract_tag(raw_text, "evidence")
            if not grounded_answer or not evidence:
                raise ValueError("Step 3 returned empty grounded_answer or evidence.")
            outputs["grounded_answer"] = grounded_answer
            outputs["evidence"] = evidence

        elif job.step == 4:
            prompt_template = self.prompts["choices_prompt"]
            raw_text = await self._infer_text_async(
                prompt_template.format(
                    question=outputs.get("question", ""),
                    blind_answer=outputs.get("blind_answer", ""),
                    blind_confidence=outputs.get("blind_confidence", ""),
                    grounded_answer=outputs.get("grounded_answer", ""),
                    evidence=outputs.get("evidence", ""),
                ),
                step=job.step,
            )
            outputs.update(self._parse_choices_output(raw_text))

        else:
            raise ValueError(f"Unsupported pipeline step: {job.step}")

        return RLQAPipelineJob(
            item_id=job.item_id,
            step=job.step + 1,
            payload=payload,
            previous_outputs=outputs,
        )

    def _final_result(self, item_id: int, payload: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        choices = outputs["choices"]
        correct_label = outputs["correct_label"]
        correct_answer = outputs["correct_answer"]
        question = outputs["question"]
        prompt_text = self._format_multiple_choice_prompt(question, choices)
        result = {
            "item_id": item_id,
            "id": payload.get("id", ""),
            "chunk_index": payload.get("chunk_index"),
            "title": payload.get("title", ""),
            "question": question,
            "choices": choices,
            "correct_label": correct_label,
            "correct_answer": correct_answer,
            "blind_answer": outputs.get("blind_answer", ""),
            "blind_confidence": outputs.get("blind_confidence", ""),
            "grounded_answer": outputs.get("grounded_answer", ""),
            "evidence": outputs.get("evidence", ""),
            "difficulty": outputs.get("difficulty", "unknown"),
            "rl_suitability": outputs.get("rl_suitability", "rejected"),
            "rejection_reason": outputs.get("rejection_reason", ""),
            "qa_generator": self.inference_config.get("MODEL_NAME", ""),
            "messages": [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": f"{correct_label}. {correct_answer}"},
            ],
            "cache_key": payload.get("cache_key", ""),
        }
        return result

    def _parse_choices_output(self, raw_text: str) -> Dict[str, Any]:
        choices = []
        for label in CHOICE_LABELS:
            choice_text = self._extract_tag(raw_text, f"choice_{label.lower()}")
            if not choice_text:
                raise ValueError(f"Step 4 returned empty choice_{label.lower()}.")
            choices.append({"label": label, "text": choice_text})

        correct_label = self._extract_tag(raw_text, "correct_label").strip().upper()
        correct_answer = self._extract_tag(raw_text, "correct_answer")
        difficulty = self._normalize_choice_value(
            self._extract_tag(raw_text, "difficulty"),
            DIFFICULTY_VALUES,
            "unknown",
        )
        rl_suitability = self._normalize_choice_value(
            self._extract_tag(raw_text, "rl_suitability"),
            SUITABILITY_VALUES,
            "rejected",
        )
        rejection_reason = self._extract_tag(raw_text, "rejection_reason")

        if correct_label not in CHOICE_LABELS:
            raise ValueError("Step 4 returned invalid correct_label.")
        matching_choices = [choice for choice in choices if choice["label"] == correct_label]
        if len(matching_choices) != 1:
            raise ValueError("Step 4 did not produce exactly one correct label.")
        normalized_choice_texts = [choice["text"].strip() for choice in choices]
        if len(set(normalized_choice_texts)) != len(normalized_choice_texts):
            raise ValueError("Step 4 returned duplicate choices.")
        if not correct_answer:
            correct_answer = matching_choices[0]["text"]
        if matching_choices[0]["text"].strip() != correct_answer.strip():
            raise ValueError("Step 4 correct_answer does not match the correct choice text.")

        return {
            "choices": choices,
            "correct_label": correct_label,
            "correct_answer": correct_answer,
            "difficulty": difficulty,
            "rl_suitability": rl_suitability,
            "rejection_reason": rejection_reason,
        }

    def _format_multiple_choice_prompt(self, question: str, choices: List[Dict[str, str]]) -> str:
        choice_lines = "\n".join(f"{choice['label']}. {choice['text']}" for choice in choices)
        return f"{question}\n\n{choice_lines}\n\n正しい選択肢をA、B、C、Dから1つ選んでください。"

    def _validate_prompts(self) -> None:
        required_keys = {
            "question_prompt",
            "blind_answer_prompt",
            "grounded_answer_prompt",
            "choices_prompt",
        }
        missing = sorted(key for key in required_keys if not self.prompts.get(key))
        if missing:
            raise ValueError(f"Missing prompt settings: {', '.join(missing)}")

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
                        f"Retrying RL QA inference ({attempt + 1}/{self.max_retries}) "
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
            1: ("1", "step1", "step_1", "question", "question_prompt"),
            2: ("2", "step2", "step_2", "blind", "blind_answer", "blind_answer_prompt"),
            3: ("3", "step3", "step_3", "grounded", "grounded_answer", "grounded_answer_prompt"),
            4: ("4", "step4", "step_4", "choices", "choices_prompt"),
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

    def _normalize_choice_value(self, value: str, allowed_values: set[str], default: str) -> str:
        normalized = value.strip().lower()
        return normalized if normalized in allowed_values else default

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

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
