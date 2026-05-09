import json
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import httpx
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

from commons.utils_msg import msg_debug, msg_info


class CPTDatasetPipeline:
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
                "SERVER_URL": server_url,
                "MODEL_NAME": model_name,
            }
        )
        self.client = OpenAI(
            base_url=self.inference_config.get("SERVER_URL"),
            api_key=self.inference_config.get("API_KEY"),
            timeout=self.inference_config.get("timeout", 600),
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
        self.save_batch_size = int(settings.get("save_batch_size", self.batch_size))
        self.max_retries = int(settings.get("max_retries", 3))
        self.wait_seconds = float(settings.get("wait_seconds", 5))
        self.copyright_mitigation = bool(settings.get("copyright_mitigation", False))
        self.keep_intermediate = bool(settings.get("keep_intermediate", False))
        self.prompts = self._load_prompts(settings.get("prompts", []))

        if self.max_chars <= 0:
            raise ValueError("max_chars must be greater than 0.")
        if self.overlap_chars < 0 or self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be 0 or greater and smaller than max_chars.")
        if not 0.0 < self.train_ratio <= 1.0:
            raise ValueError("train_ratio must be greater than 0.0 and less than or equal to 1.0.")

    def load_entries(self, source_path: Path) -> List[Dict]:
        entries: List[Dict] = []
        for file_path in self._collect_source_files(source_path):
            entries.extend(self._load_file(file_path))
        return entries

    def build_dataset(self, source_path: Path) -> Tuple[Path, Path | None, Dict[str, int]]:
        entries = self.load_entries(source_path)
        chunks = self._build_and_save_chunks(entries)

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
        }
        self._write_stats(stats)
        return train_path, validation_output_path, stats

    def _build_and_save_chunks(self, entries: List[Dict]) -> List[Dict]:
        all_chunks_path = self.output_dir / "all.jsonl"
        cache_path = self.output_dir / "cache_processed_ids.txt"
        batch_status_path = self.output_dir / "batch_status.jsonl"
        all_chunks_path.touch(exist_ok=True)
        cache_path.touch(exist_ok=True)
        batch_status_path.touch(exist_ok=True)
        processed_ids = self._load_processed_ids(cache_path)

        if processed_ids and all_chunks_path.exists():
            print(msg_info(f"Loaded {len(processed_ids)} processed IDs from cache."))

        chunks = self._load_jsonl_records(all_chunks_path) if all_chunks_path.exists() else []
        pending_entries = [entry for entry in entries if self._entry_cache_id(entry) not in processed_ids]

        for index, entry in enumerate(pending_entries, start=1):
            entry_id = self._entry_cache_id(entry)
            print(msg_info(f"CPT row {index} / {len(pending_entries)} id={entry_id}"))

            batch_chunks = self._build_chunks([entry])
            self._append_jsonl_record(
                batch_status_path,
                {
                    "row": index,
                    "input_records": 1,
                    "output_chunks": len(batch_chunks),
                    "cached_ids": [entry_id],
                },
            )
            print(msg_info(f"Saved CPT batch status. output_chunks={len(batch_chunks)}"))
            self._append_jsonl_records(all_chunks_path, batch_chunks)
            self._append_processed_ids(cache_path, [entry_id])
            chunks.extend(batch_chunks)

        return chunks

    def _load_prompts(self, prompts_settings: List[Dict]) -> Dict[str, str]:
        prompts_dict: Dict[str, str] = {}
        for prompt_path_dict in prompts_settings:
            key, prompt_path = list(prompt_path_dict.items())[0]
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompts_dict[key] = f.read()
        return prompts_dict

    def _infer_text(self, prompt: str) -> str:
        user_content = [{"type": "text", "text": prompt}]
        max_tokens = int(self.inference_config.get("max_tokens", 2048))
        temperature = self.inference_config.get("temperature", 0)
        top_p = self.inference_config.get("top_p", 1.0)

        last_exc = None
        for i in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.inference_config.get("MODEL_NAME"),
                    messages=[{"role": "user", "content": user_content}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                response_content = response.choices[0].message.content
                if response_content is None:
                    last_exc = RuntimeError("Model returned empty content (None).")
                    raise last_exc
                if isinstance(response_content, str):
                    normalized = response_content.strip()
                elif isinstance(response_content, list):
                    normalized = "".join(
                        str(part.get("text", "")) if isinstance(part, dict) else str(part)
                        for part in response_content
                    ).strip()
                else:
                    normalized = str(response_content).strip()
                if normalized:
                    return normalized
                last_exc = RuntimeError("Model returned blank text.")
            except Exception as exc:
                last_exc = exc
                if i < self.max_retries - 1:
                    sleep = min((self.wait_seconds * (2 ** i)) + random.random() * 0.2, 30.0)
                    time.sleep(sleep)
                    if isinstance(exc, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError)):
                        print(msg_debug(f"Retrying due to network error. Attempt {i + 1}/{self.max_retries}."))
                    continue
        print(msg_debug(f"Inference failed after {self.max_retries} attempts. last_error={last_exc}"))
        return ""

    def _infer_texts(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        max_workers = max(1, min(self.batch_size, len(prompts)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._infer_text, prompts))

    def _reconstruct_texts(self, texts: List[str]) -> List[Tuple[str, str, str]]:
        if not texts:
            return []

        to_bullet_prompt = self.prompts.get("to_bullet_points_prompt")
        to_text_prompt = self.prompts.get("bullet_points_to_text_prompt")
        if not to_bullet_prompt or not to_text_prompt:
            raise ValueError(
                "to_bullet_points_prompt and bullet_points_to_text_prompt must be set when copyright_mitigation is true."
            )

        print(msg_info("now generating BULLET POINTS..."))
        bullet_prompts = [to_bullet_prompt.format(text=text) for text in texts]
        bullet_texts = self._infer_texts(bullet_prompts)

        print(msg_info("now reconstructing TEXT..."))
        reconstruct_prompts = [to_text_prompt.format(text=bullet_text) for bullet_text in bullet_texts]
        reconstructed_texts = self._infer_texts(reconstruct_prompts)

        results: List[Tuple[str, str, str]] = []
        for original_text, bullet_text, reconstructed_text in zip(texts, bullet_texts, reconstructed_texts):
            if reconstructed_text and reconstructed_text.strip() != "|||":
                results.append((reconstructed_text.strip(), bullet_text.strip(), original_text))
            else:
                results.append(("", bullet_text.strip(), original_text))
        return results

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

    def _build_chunks(self, entries: List[Dict]) -> List[Dict]:
        chunks: List[Dict] = []
        seen_texts = set()

        for entry in entries:
            text = entry.get(self.target_key)
            if not isinstance(text, str):
                continue

            normalized = self._normalize_text(text)
            if not normalized or len(normalized) < self.min_chars:
                continue

            title = str(entry.get(self.title_key, "")).strip()
            if self.include_title and title:
                normalized = f"{title}\n\n{normalized}"

            original_chunks = self._split_text(normalized)
            if self.copyright_mitigation:
                reconstructed_chunks = self._reconstruct_texts(original_chunks)
            else:
                reconstructed_chunks = [(chunk_text, "", chunk_text) for chunk_text in original_chunks]

            for chunk_index, (chunk_text, bullet_text, original_text) in enumerate(reconstructed_chunks):
                if len(chunk_text) < self.min_chars:
                    continue
                if self.deduplicate:
                    if chunk_text in seen_texts:
                        continue
                    seen_texts.add(chunk_text)
                chunks.append(
                    {
                        self.text_key: chunk_text,
                        "id": str(entry.get(self.id_key, "")),
                        "title": title,
                        "source_file": str(entry.get("source_file", "")),
                        "chunk_index": chunk_index,
                        "copyright_mitigation": self.copyright_mitigation,
                    }
                )
                if self.copyright_mitigation:
                    chunks[-1]["cpt_generator"] = self.inference_config.get("MODEL_NAME", "")
                if self.keep_intermediate:
                    chunks[-1]["bullet_points"] = bullet_text
                    chunks[-1]["original_text"] = original_text

        return chunks

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
