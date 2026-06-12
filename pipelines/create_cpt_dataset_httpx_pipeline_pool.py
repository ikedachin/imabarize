import asyncio
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

from commons.utils_msg import msg_debug, msg_error, msg_info


RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class CPTCandidateResult:
    candidate: Dict[str, Any]
    chunk: Optional[Dict[str, Any]] = None
    failure: Optional[Dict[str, Any]] = None


class CPTDatasetPipelinePool:
    def __init__(self, settings: Dict):
        self.settings = settings
        self.inference_config = dict(settings.get("infer_config", {}))
        self.output_dir = Path(settings.get("output_path", "./test_output/cpt")).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

        self.target_key = settings.get("target_key", "content")
        self.title_key = settings.get("title_key", "title")
        self.id_key = settings.get("id_key", "id")
        self.text_key = settings.get("text_key", "text")
        self.include_title = bool(settings.get("include_title", True))
        self.min_chars = int(settings.get("min_chars", 128))
        self.max_chars = int(settings.get("max_chars", 4096))
        self.overlap_chars = int(settings.get("overlap_chars", 256))
        self.train_ratio = float(settings.get("train_ratio", 0.98))
        self.seed = int(settings.get("seed", 42))
        self.shuffle = bool(settings.get("shuffle", True))
        self.deduplicate = bool(settings.get("deduplicate", True))
        self.batch_size = int(settings.get("batch_size", 4))
        self.max_in_flight = max(1, int(settings.get("max_in_flight", self.batch_size)))
        self.pipeline_batch_size = int(
            settings.get("pipeline_batch_size", max(self.batch_size, self.max_in_flight * 4))
        )
        self.max_retries = int(settings.get("max_retries", 3))
        self.wait_seconds = float(settings.get("wait_seconds", 5))
        self.retry_jitter_seconds = float(settings.get("retry_jitter_seconds", 0.2))
        self.retry_max_delay = float(settings.get("retry_max_delay", 30.0))
        self.copyright_mitigation = bool(settings.get("copyright_mitigation", False))
        self.copyright_mitigation_failure_policy = str(
            settings.get("copyright_mitigation_failure_policy", "fail")
        ).strip().lower()
        self.keep_intermediate = bool(settings.get("keep_intermediate", False))
        self.cpt_enable_thinking = self._parse_optional_bool(settings.get("cpt_enable_thinking"))
        self.prompts = self._load_prompts(settings.get("prompts", []))

        if self.max_chars <= 0:
            raise ValueError("max_chars must be greater than 0.")
        if self.overlap_chars < 0 or self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be 0 or greater and smaller than max_chars.")
        if not 0.0 < self.train_ratio <= 1.0:
            raise ValueError("train_ratio must be greater than 0.0 and less than or equal to 1.0.")

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

    def _uses_original_fallback(self) -> bool:
        return self.copyright_mitigation_failure_policy in {"original", "use_original", "fallback_original"}

    async def aclose(self) -> None:
        await self.client.aclose()

    def load_entries(self, source_path: Path) -> List[Dict]:
        entries: List[Dict] = []
        for file_path in self._collect_source_files(source_path):
            entries.extend(self._load_file(file_path))
        return entries

    async def build_dataset(self, source_path: Path) -> Tuple[Path, Path | None, Dict[str, int]]:
        entries = self.load_entries(source_path)
        chunks = await self._build_and_save_chunks_async(entries)

        if self.shuffle:
            random.Random(self.seed).shuffle(chunks)

        split_index = int(len(chunks) * self.train_ratio)
        if len(chunks) > 1 and split_index == len(chunks):
            split_index = len(chunks) - 1

        train_chunks = chunks[:split_index]
        validation_chunks = chunks[split_index:]

        train_path = self.output_dir / "train.jsonl"
        validation_path = self.output_dir / "validation.jsonl"
        self._write_jsonl(train_path, train_chunks)
        validation_output_path = None
        if validation_chunks:
            self._write_jsonl(validation_path, validation_chunks)
            validation_output_path = validation_path

        stats = {
            "entries": len(entries),
            "chunks": len(chunks),
            "train_chunks": len(train_chunks),
            "validation_chunks": len(validation_chunks),
            "failed_chunks": self._count_jsonl_records(self.output_dir / "all.failures.jsonl"),
        }
        self._write_stats(stats)
        return train_path, validation_output_path, stats

    async def _build_and_save_chunks_async(self, entries: List[Dict]) -> List[Dict]:
        all_chunks_path = self.output_dir / "all.jsonl"
        failures_path = self.output_dir / "all.failures.jsonl"
        cache_path = self.output_dir / "cache_processed_ids.txt"
        batch_status_path = self.output_dir / "batch_status.jsonl"
        all_chunks_path.touch(exist_ok=True)
        cache_path.touch(exist_ok=True)
        batch_status_path.touch(exist_ok=True)
        processed_ids = self._load_processed_ids(cache_path)

        if processed_ids:
            print(msg_info(f"Loaded {len(processed_ids)} processed IDs from cache."))

        chunks = self._load_jsonl_records(all_chunks_path) if all_chunks_path.exists() else []
        pending_entries = [entry for entry in entries if self._entry_cache_id(entry) not in processed_ids]

        print(msg_info(f"Pre-chunking {len(pending_entries)} pending entries..."))
        chunk_candidates = self._prepare_chunk_candidates(pending_entries)
        print(msg_info(f"Prepared {len(chunk_candidates)} chunk candidates."))

        if not chunk_candidates:
            return chunks

        entry_candidate_counts: Dict[str, int] = {}
        for candidate in chunk_candidates:
            eid = candidate["_entry_id"]
            entry_candidate_counts[eid] = entry_candidate_counts.get(eid, 0) + 1

        entry_processed_counts: Dict[str, int] = {eid: 0 for eid in entry_candidate_counts}
        entries_with_output: set[str] = set()
        entries_with_failure: set[str] = set()
        already_cached: set[str] = set()
        seen_texts = {str(chunk.get(self.text_key, "")) for chunk in chunks if chunk.get(self.text_key)}
        output_lock = asyncio.Lock()
        status_counter = 0
        saved_count = 0
        failed_count = 0
        start_time = time.monotonic()

        queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
        for candidate in chunk_candidates:
            await queue.put(candidate)

        worker_count = max(1, min(self.max_in_flight, len(chunk_candidates)))
        print(
            msg_info(
                "Generating CPT with pipeline-pool async httpx pipeline: "
                f"candidates={len(chunk_candidates)}, workers={worker_count}, "
                f"max_in_flight={self.max_in_flight}, input_window_hint={self.pipeline_batch_size}"
            )
        )

        async def _handle_result(result: CPTCandidateResult) -> None:
            nonlocal failed_count, saved_count, status_counter
            candidate = result.candidate
            eid = candidate["_entry_id"]
            chunk = result.chunk
            failure = result.failure
            save_chunks: List[Dict[str, Any]] = []
            newly_cached: List[str] = []

            async with output_lock:
                entry_processed_counts[eid] += 1

                if failure is not None:
                    failed_count += 1
                    entries_with_failure.add(eid)
                    self._append_jsonl_record(failures_path, failure)
                elif chunk is not None:
                    public_chunk = {k: v for k, v in chunk.items() if not k.startswith("_")}
                    chunk_text = str(public_chunk.get(self.text_key, ""))
                    if not self.deduplicate or chunk_text not in seen_texts:
                        if chunk_text:
                            seen_texts.add(chunk_text)
                        save_chunks.append(public_chunk)
                        saved_count += 1
                        entries_with_output.add(eid)

                if (
                    entry_processed_counts[eid] >= entry_candidate_counts[eid]
                    and eid in entries_with_output
                    and eid not in entries_with_failure
                    and eid not in already_cached
                ):
                    newly_cached.append(eid)
                    already_cached.add(eid)

                self._append_jsonl_records(all_chunks_path, save_chunks)
                if newly_cached:
                    self._append_processed_ids(cache_path, newly_cached)

                status_counter += 1
                self._append_jsonl_record(
                    batch_status_path,
                    {
                        "batch": status_counter,
                        "input_records": 1,
                        "output_chunks": len(save_chunks),
                        "failed_chunks": 1 if failure is not None else 0,
                        "cached_ids": newly_cached,
                        "entry_id": eid,
                        "chunk_index": candidate["_chunk_index"],
                    },
                )

        async def _worker(worker_id: int) -> None:
            while True:
                candidate = await queue.get()
                if candidate is None:
                    queue.task_done()
                    return

                try:
                    result = await self._process_candidate_async(candidate)
                    await _handle_result(result)
                    print(
                        msg_info(
                            f"CPT candidate complete worker_id={worker_id} "
                            f"id={candidate['_entry_id']} chunk_index={candidate['_chunk_index']} "
                            f"saved={saved_count} failed={failed_count} "
                            f"in_flight={self.current_in_flight}"
                        )
                    )
                except Exception as exc:
                    failure = self._failure_record(candidate, "candidate", str(exc))
                    await _handle_result(CPTCandidateResult(candidate=candidate, failure=failure))
                    print(
                        msg_error(
                            f"CPT candidate failed worker_id={worker_id} "
                            f"id={candidate['_entry_id']} chunk_index={candidate['_chunk_index']} "
                            f"error={exc}"
                        )
                    )
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(_worker(worker_id)) for worker_id in range(worker_count)]
        await queue.join()
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

        for eid in entry_candidate_counts:
            if eid in entries_with_failure:
                print(msg_info(f"Skipped cache for id={eid} because one or more chunks failed."))
            elif eid not in entries_with_output:
                print(msg_info(f"Skipped cache for id={eid} because no chunks were generated."))

        total_elapsed = time.monotonic() - start_time
        print(
            msg_info(
                f"CPT pipeline-pool finished candidates={len(chunk_candidates)} "
                f"saved={saved_count} failed={failed_count} elapsed={total_elapsed:.2f}s "
                f"max_observed_in_flight={self.max_observed_in_flight}"
            )
        )
        return chunks + self._load_jsonl_records(all_chunks_path)[len(chunks) :]

    async def _process_candidate_async(self, candidate: Dict[str, Any]) -> CPTCandidateResult:
        original_text = candidate["_original_text"]
        bullet_text = ""
        chunk_text = original_text
        used_original_fallback = False

        if self.copyright_mitigation:
            try:
                chunk_text, bullet_text, _ = await self._reconstruct_text_async(original_text)
            except Exception as exc:
                if self._uses_original_fallback():
                    chunk_text = original_text
                    bullet_text = ""
                    used_original_fallback = True
                else:
                    failed_step = "to_bullet_points"
                    if "reconstruction" in str(exc).lower():
                        failed_step = "bullet_points_to_text"
                    return CPTCandidateResult(
                        candidate=candidate,
                        failure=self._failure_record(candidate, failed_step, str(exc)),
                    )

        if (not chunk_text or chunk_text.strip() == "|||") and self._uses_original_fallback():
            chunk_text = original_text
            used_original_fallback = True

        if not chunk_text or chunk_text.strip() == "|||" or len(chunk_text) < self.min_chars:
            if self.copyright_mitigation and not self._uses_original_fallback():
                failed_step = "to_bullet_points"
                return CPTCandidateResult(
                    candidate=candidate,
                    failure=self._failure_record(candidate, failed_step, "Generated text was empty or too short."),
                )
            return CPTCandidateResult(candidate=candidate)

        entry = candidate["_entry"]
        mitigation_applied = self.copyright_mitigation and not used_original_fallback
        chunk: Dict[str, Any] = {
            self.text_key: chunk_text.strip(),
            "id": str(entry.get(self.id_key, "")),
            "title": candidate["_title"],
            "source_file": str(entry.get("source_file", "")),
            "chunk_index": candidate["_chunk_index"],
            "copyright_mitigation": mitigation_applied,
            "_entry_id": candidate["_entry_id"],
        }
        if mitigation_applied:
            chunk["cpt_generator"] = self.inference_config.get("MODEL_NAME", "")
        if self.keep_intermediate:
            chunk["bullet_points"] = bullet_text
            chunk["original_text"] = original_text
        return CPTCandidateResult(candidate=candidate, chunk=chunk)

    async def _reconstruct_text_async(self, text: str) -> Tuple[str, str, str]:
        to_bullet_prompt = self.prompts.get("to_bullet_points_prompt")
        to_text_prompt = self.prompts.get("bullet_points_to_text_prompt")
        if not to_bullet_prompt or not to_text_prompt:
            raise ValueError(
                "to_bullet_points_prompt and bullet_points_to_text_prompt must be set when copyright_mitigation is true."
            )

        bullet_text = await self._infer_text_async(to_bullet_prompt.format(text=text), step="to_bullet_points")
        if not bullet_text:
            raise ValueError("Bullet point generation returned blank text.")
        reconstructed_text = await self._infer_text_async(
            to_text_prompt.format(text=bullet_text),
            step="bullet_points_to_text",
        )
        if reconstructed_text and reconstructed_text.strip() != "|||":
            return reconstructed_text.strip(), bullet_text.strip(), text
        if not reconstructed_text:
            raise ValueError("Text reconstruction returned blank text.")
        return "", bullet_text.strip(), text

    def _chat_payload(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.inference_config.get("MODEL_NAME"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(self.inference_config.get("max_tokens", 2048)),
            "temperature": self.inference_config.get("temperature", 0),
            "top_p": self.inference_config.get("top_p", 1.0),
            "stream": False,
        }
        if self.cpt_enable_thinking is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.cpt_enable_thinking,
            }
        return payload

    async def _post_chat_completion(self, payload: Dict[str, Any]) -> httpx.Response:
        return await self.client.post("/chat/completions", json=payload)

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

    async def _infer_text_async(self, prompt: str, step: str) -> str:
        payload = self._chat_payload(prompt)
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
                raise ValueError(f"{step} returned blank text.")

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
                print(msg_debug(f"Retrying CPT inference ({attempt + 1}/{self.max_retries}) after {delay:.2f}s. reason={reason}"))
                await asyncio.sleep(delay)

        print(msg_error(f"Inference failed after {self.max_retries} attempts. last_error={last_exc}"))
        return ""

    def _failure_record(self, candidate: Dict[str, Any], failed_step: str, error: str) -> Dict[str, Any]:
        entry = candidate["_entry"]
        return {
            "id": str(entry.get(self.id_key, "")),
            "entry_id": candidate["_entry_id"],
            "chunk_index": candidate["_chunk_index"],
            "source_file": str(entry.get("source_file", "")),
            "failed": True,
            "failed_step": failed_step,
            "error": error,
            "original_text": candidate["_original_text"],
            "cpt_generator": self.inference_config.get("MODEL_NAME", ""),
        }

    def _prepare_chunk_candidates(self, entries: List[Dict]) -> List[Dict]:
        candidates: List[Dict] = []
        for entry in entries:
            entry_id = self._entry_cache_id(entry)
            text = entry.get(self.target_key)
            if not isinstance(text, str):
                continue

            normalized = self._normalize_text(text)
            if not normalized or len(normalized) < self.min_chars:
                continue

            title = str(entry.get(self.title_key, "")).strip()
            if self.include_title and title:
                normalized = f"{title}\n\n{normalized}"

            for chunk_index, chunk_text in enumerate(self._split_text(normalized)):
                candidates.append(
                    {
                        "_entry_id": entry_id,
                        "_entry": entry,
                        "_title": title,
                        "_chunk_index": chunk_index,
                        "_original_text": chunk_text,
                    }
                )
        return candidates

    def _load_prompts(self, prompts_settings: List[Dict]) -> Dict[str, str]:
        prompts_dict: Dict[str, str] = {}
        for prompt_path_dict in prompts_settings:
            key, prompt_path = list(prompt_path_dict.items())[0]
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompts_dict[key] = f.read()
        return prompts_dict

    def _collect_source_files(self, source_path: Path) -> List[Path]:
        if source_path.is_file():
            return [source_path]
        if source_path.is_dir():
            return sorted(
                p
                for p in source_path.rglob("*")
                if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".txt", ".md"}
            )
        raise FileNotFoundError(f"Source path not found: {source_path}")

    def _load_file(self, file_path: Path) -> List[Dict]:
        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            return list(self._load_jsonl(file_path))
        if suffix == ".json":
            return self._load_json(file_path)
        if suffix in {".txt", ".md"}:
            return [
                {
                    self.id_key: file_path.stem,
                    self.title_key: file_path.stem,
                    self.target_key: file_path.read_text(encoding="utf-8"),
                    "source_file": str(file_path),
                }
            ]
        print(msg_debug(f"Skipping unsupported file: {file_path}"))
        return []

    def _load_jsonl(self, file_path: Path) -> Iterable[Dict]:
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

    def _load_json(self, file_path: Path) -> List[Dict]:
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

    def _load_jsonl_records(self, file_path: Path) -> List[Dict]:
        if not file_path.exists():
            return []
        records: List[Dict] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
        return records

    def _count_jsonl_records(self, file_path: Path) -> int:
        return len(self._load_jsonl_records(file_path))

    def _entry_cache_id(self, entry: Dict) -> str:
        entry_id = entry.get(self.id_key)
        if entry_id is not None:
            return str(entry_id)
        source_file = str(entry.get("source_file", ""))
        title = str(entry.get(self.title_key, ""))
        text = str(entry.get(self.target_key, ""))
        return f"{source_file}:{title}:{hash(text)}"

    def _load_processed_ids(self, cache_path: Path) -> set[str]:
        if not cache_path.exists():
            return set()
        with open(cache_path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())

    def _append_processed_ids(self, cache_path: Path, entry_ids: List[str]) -> None:
        if not entry_ids:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a", encoding="utf-8") as f:
            for entry_id in entry_ids:
                f.write(f"{entry_id}\n")

    def _append_jsonl_records(self, save_path: Path, records: List[Dict]) -> None:
        if not records:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.touch(exist_ok=True)
            return
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "a", encoding="utf-8") as f:
            for record in records:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        print(msg_info(f"Appended {len(records)} records to: {save_path}"))

    def _append_jsonl_record(self, save_path: Path, record: Dict) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_text(self, text: str) -> List[str]:
        if len(text) <= self.max_chars:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            if end < len(text):
                boundary = self._find_boundary(text, start, end)
                if boundary > start:
                    end = boundary

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(end - self.overlap_chars, start + 1)
        return chunks

    def _find_boundary(self, text: str, start: int, end: int) -> int:
        search_start = start + int(self.max_chars * 0.5)
        for separator in ("\n\n", "\n", "。", "、"):
            boundary = text.rfind(separator, search_start, end)
            if boundary != -1:
                return boundary + len(separator)
        return end

    def _write_jsonl(self, save_path: Path, records: List[Dict]) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for record in records:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        print(msg_info(f"Saved {len(records)} records to: {save_path}"))

    def _write_stats(self, stats: Dict[str, int]) -> None:
        save_path = self.output_dir / "stats.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
