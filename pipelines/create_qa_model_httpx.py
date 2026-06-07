import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

from commons.utils_msg import msg_debug, msg_error, msg_info


RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
NUM_STEPS = 5


@dataclass
class PipelineJob:
    item_id: int
    step: int
    payload: Dict[str, Any]
    previous_outputs: Dict[str, str] = field(default_factory=dict)


class QAPipeline:
    def __init__(self, settings: Dict):
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

        self.output_dir = (
            Path(settings.get("output_path", "./json_output/qa"))
            .expanduser()
            .resolve()
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        configured_batch_size = int(settings.get("batch_size", 1))
        self.max_in_flight = int(settings.get("max_in_flight", configured_batch_size))
        self.max_in_flight = max(1, self.max_in_flight)
        self.request_semaphore = asyncio.Semaphore(self.max_in_flight)
        self._in_flight_lock = asyncio.Lock()
        self.current_in_flight = 0
        self.max_observed_in_flight = 0

        self.prompts = self._load_prompts(settings.get("prompts", []))
        self.max_retries = int(settings.get("max_retries", 8))
        self.wait_seconds = float(settings.get("wait_seconds", 0.25))
        self.retry_jitter_seconds = float(settings.get("retry_jitter_seconds", self.wait_seconds))
        self.retry_max_delay = float(settings.get("retry_max_delay", 30.0))
        self.thinking_enabled_by_step = self._load_thinking_enabled_by_step(
            settings.get("thinking_enabled_by_step", {})
        )

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

    def _load_prompts(self, prompts_settings: List[Dict]) -> Dict[str, str]:
        prompts_dict: Dict[str, str] = {}
        for prompt_path_dict in prompts_settings:
            key, prompt_path = list(prompt_path_dict.items())[0]
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompts_dict[key] = f.read()
        return prompts_dict

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

    def _load_thinking_enabled_by_step(self, raw_settings: Dict[str, Any]) -> Dict[int, Optional[bool]]:
        step_aliases = {
            1: ("1", "step1", "step_1", "question", "question_prompt"),
            2: ("2", "step2", "step_2", "answer", "answer_prompt"),
            3: ("3", "step3", "step_3", "thinking", "thinking_prompt"),
            4: ("4", "step4", "step_4", "refine", "refine_answer", "refine_answer_prompt"),
            5: ("5", "step5", "step_5", "eval", "eval_prompt"),
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

    async def aclose(self) -> None:
        await self.client.aclose()

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
        if (not text) or (not isinstance(text, str)):
            return ""
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        if (start_tag in text) and (end_tag in text):
            text = text.split(start_tag)[-1]
            text = text.split(end_tag)[0]
        elif start_tag in text:
            text = text.split(start_tag)[-1]
        elif end_tag in text:
            text = text.split(end_tag)[0]
        return text.strip()

    def _chat_payload(self, prompt: str, step: Optional[int] = None) -> Dict:
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

    async def _increment_in_flight(self) -> int:
        async with self._in_flight_lock:
            self.current_in_flight += 1
            self.max_observed_in_flight = max(
                self.max_observed_in_flight,
                self.current_in_flight,
            )
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
                        f"Retrying inference ({attempt + 1}/{self.max_retries}) "
                        f"after {delay:.2f}s. reason={reason}"
                    )
                )
                await asyncio.sleep(delay)

        print(msg_error(f"Inference failed after {self.max_retries} attempts. last_error={last_exc}"))
        return ""

    def _strip_tag_block(self, text: str, tag: str) -> str:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        if start_tag not in text:
            return text.strip()
        before, _, rest = text.partition(start_tag)
        _, _, after = rest.partition(end_tag)
        return f"{before}{after}".strip()

    def _extract_tag_if_present(self, text: str, tag: str) -> str:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        if start_tag not in text and end_tag not in text:
            return ""
        return self._extract_tag(text, tag)

    def _final_result(self, item_id: int, outputs: Dict[str, str]) -> Dict[str, Any]:
        thinking_text = outputs.get("thinking", "")
        answer_text = outputs.get("answer", "")
        question_text = outputs.get("question", "")
        return {
            "item_id": item_id,
            "question": question_text,
            "thinking": thinking_text,
            "answer": answer_text,
            "eval": outputs.get("eval", ""),
            "qa_generator": self.inference_config.get("MODEL_NAME", ""),
            "messages": [
                {
                    "role": "user",
                    "content": question_text,
                },
                {
                    "role": "assistant",
                    "content": f"<think>{thinking_text}</think>\n\n{answer_text}",
                },
            ],
        }

    def _failure_result(
        self,
        item_id: int,
        step: int,
        error: str,
        outputs: Dict[str, str],
    ) -> Dict[str, Any]:
        return {
            "item_id": item_id,
            "failed": True,
            "failed_step": step,
            "error": error,
            "previous_outputs": outputs,
            "qa_generator": self.inference_config.get("MODEL_NAME", ""),
        }

    async def _run_one_step(self, job: PipelineJob) -> PipelineJob:
        text = str(job.payload.get("text", ""))
        outputs = dict(job.previous_outputs)

        if job.step == 1:
            prompt_template = self.prompts.get("question_prompt")
            if not prompt_template:
                raise ValueError("question_prompt must be set in prompts.")
            raw_text = await self._infer_text_async(
                prompt_template.format(
                    text=text,
                    random_token=job.payload.get("random_token", ""),
                ),
                step=job.step,
            )
            question_text = self._extract_tag(raw_text, "question")
            if not question_text:
                raise ValueError("Step 1 returned empty question.")
            outputs["question"] = question_text

        elif job.step == 2:
            prompt_template = self.prompts.get("answer_prompt")
            if not prompt_template:
                raise ValueError("answer_prompt must be set in prompts.")
            raw_text = await self._infer_text_async(
                prompt_template.format(text=text, question=outputs.get("question", "")),
                step=job.step,
            )
            answer_text = self._extract_tag(raw_text, "answer")
            if not answer_text:
                raise ValueError("Step 2 returned empty answer.")
            outputs["answer"] = answer_text

        elif job.step == 3:
            prompt_template = self.prompts.get("thinking_prompt")
            if prompt_template:
                raw_text = await self._infer_text_async(
                    prompt_template.format(
                        text=text,
                        question=outputs.get("question", ""),
                        answer=outputs.get("answer", ""),
                    ),
                    step=job.step,
                )
                thinking_text = self._extract_tag(raw_text, "think") or raw_text.strip()
                if not thinking_text:
                    raise ValueError("Step 3 returned empty thinking.")
                outputs["thinking"] = thinking_text
            else:
                outputs.setdefault("thinking", "")

        elif job.step == 4:
            prompt_template = self.prompts.get("refine_answer_prompt")
            if prompt_template:
                current_answer = (
                    f"<think>{outputs.get('thinking', '')}</think>\n\n"
                    f"{outputs.get('answer', '')}"
                )
                raw_text = await self._infer_text_async(
                    prompt_template.format(
                        text=text,
                        question=outputs.get("question", ""),
                        answer=current_answer,
                    ),
                    step=job.step,
                )
                refined_thinking = self._extract_tag(raw_text, "think")
                refined_answer = self._extract_tag_if_present(raw_text, "answer")
                if not refined_answer:
                    refined_answer = self._strip_tag_block(raw_text, "think")
                if not refined_answer:
                    raise ValueError("Step 4 returned empty refined answer.")
                if refined_thinking:
                    outputs["thinking"] = refined_thinking
                outputs["answer"] = refined_answer

        elif job.step == 5:
            prompt_template = self.prompts.get("eval_prompt")
            if prompt_template:
                raw_text = await self._infer_text_async(
                    prompt_template.format(
                        think=outputs.get("thinking", ""),
                        answer=outputs.get("answer", ""),
                    ),
                    step=job.step,
                )
                outputs["eval"] = self._extract_tag(raw_text, "eval") or raw_text.strip()
            else:
                outputs.setdefault("eval", "")

        else:
            raise ValueError(f"Unsupported pipeline step: {job.step}")

        return PipelineJob(
            item_id=job.item_id,
            step=job.step + 1,
            payload=job.payload,
            previous_outputs=outputs,
        )

    async def create_qa_batch_async(
        self,
        texts: List[str],
        batch_size: int,
        on_result: Optional[Callable[[int, Dict[str, Any]], Awaitable[None]]] = None,
    ) -> List[Dict[str, Any]]:
        if not texts:
            return []

        question_prompt = self.prompts.get("question_prompt")
        answer_prompt = self.prompts.get("answer_prompt")

        if not question_prompt or not answer_prompt:
            raise ValueError("question_prompt and answer_prompt must be set in prompts.")

        start_time = time.monotonic()
        queue: asyncio.Queue[Optional[PipelineJob]] = asyncio.Queue()
        results: Dict[int, Dict[str, Any]] = {}
        failed_count = 0
        completed_count = 0
        next_item_id = 0
        total_items = len(texts)
        worker_count = max(1, min(self.max_in_flight, total_items))

        print(
            msg_info(
                "Generating QA with rolling async httpx pipeline: "
                f"items={total_items}, workers={worker_count}, "
                f"max_in_flight={self.max_in_flight}, input_window_hint={batch_size}"
            )
        )

        def _build_initial_job(item_id: int) -> PipelineJob:
            text = texts[item_id]
            random_token = "".join(random.sample(text, min(len(text), 10))) if text else ""
            return PipelineJob(
                item_id=item_id,
                step=1,
                payload={"text": text, "random_token": random_token},
            )

        async def _emit_result(item_id: int, result: Dict[str, Any]) -> None:
            results[item_id] = result
            if on_result is not None:
                await on_result(item_id, result)

        async def _enqueue_next_item_if_needed() -> None:
            nonlocal next_item_id
            if next_item_id < total_items:
                await queue.put(_build_initial_job(next_item_id))
                next_item_id += 1

        for _ in range(worker_count):
            await queue.put(_build_initial_job(next_item_id))
            next_item_id += 1

        async def _worker(worker_id: int) -> None:
            nonlocal completed_count, failed_count
            while True:
                job = await queue.get()
                if job is None:
                    queue.task_done()
                    return

                step_start = time.monotonic()
                print(
                    msg_debug(
                        f"Job start item_id={job.item_id} step={job.step} "
                        f"in_flight={self.current_in_flight}"
                    )
                )
                try:
                    next_job = await self._run_one_step(job)
                    elapsed = time.monotonic() - step_start
                    print(
                        msg_info(
                            f"Job complete item_id={job.item_id} step={job.step} "
                            f"elapsed={elapsed:.2f}s in_flight={self.current_in_flight} "
                            f"completed={completed_count} failed={failed_count}"
                        )
                    )

                    if job.step >= NUM_STEPS:
                        completed_count += 1
                        await _emit_result(
                            job.item_id,
                            self._final_result(job.item_id, next_job.previous_outputs),
                        )
                        await _enqueue_next_item_if_needed()
                    else:
                        await queue.put(next_job)

                except Exception as exc:
                    elapsed = time.monotonic() - step_start
                    failed_count += 1
                    print(
                        msg_error(
                            f"Job failed item_id={job.item_id} step={job.step} "
                            f"elapsed={elapsed:.2f}s error={exc} "
                            f"in_flight={self.current_in_flight} "
                            f"completed={completed_count} failed={failed_count}"
                        )
                    )
                    await _emit_result(
                        job.item_id,
                        self._failure_result(
                            item_id=job.item_id,
                            step=job.step,
                            error=str(exc),
                            outputs=job.previous_outputs,
                        ),
                    )
                    await _enqueue_next_item_if_needed()
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(_worker(worker_id)) for worker_id in range(worker_count)]
        await queue.join()
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

        total_elapsed = time.monotonic() - start_time
        print(
            msg_info(
                f"Rolling pipeline finished items={total_items} completed={completed_count} "
                f"failed={failed_count} elapsed={total_elapsed:.2f}s "
                f"max_observed_in_flight={self.max_observed_in_flight}"
            )
        )
        return [results.get(item_id, {}) for item_id in range(total_items)]

    def append_jsonl(self, save_path: Path, result: Dict) -> None:
        required_keys = ["question", "answer", "qa_generator", "messages"]
        if all(result.get(key) for key in required_keys):
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "a", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")
        else:
            print(msg_debug(f"Skipped saving due to empty required fields. result={result}"))

    def append_failure_jsonl(self, save_path: Path, result: Dict) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

    def add_cache(self, entry_id: str, cache_name: str) -> None:
        save_path = self.output_dir / f"cache_{cache_name}.txt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(f"{entry_id}\n")
