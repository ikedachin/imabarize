import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tqdm

from commons.util_settings import load_settings
from commons.utils_msg import msg_debug, msg_error, msg_info, msg_success
from pipelines.create_qa_model_httpx import QAPipeline

TEXT_EXTENSIONS = {".md", ".txt"}
JSON_EXTENSIONS = {".json", ".jsonl"}


def collect_source_files(source_path: Path) -> Tuple[List[Path], List[Path]]:
    text_files: List[Path] = []
    json_files: List[Path] = []

    if source_path.is_dir():
        for candidate in sorted(source_path.rglob("*")):
            if not candidate.is_file():
                continue
            suffix = candidate.suffix.lower()
            if suffix in TEXT_EXTENSIONS:
                text_files.append(candidate)
            elif suffix in JSON_EXTENSIONS:
                json_files.append(candidate)
        return text_files, json_files

    if source_path.is_file():
        suffix = source_path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            return [source_path], []
        if suffix in JSON_EXTENSIONS:
            return [], [source_path]
        print(msg_error(f"Unsupported file type: {suffix} for {source_path}"))
        return [], []

    print(msg_error(f"Source path not found: {source_path}"))
    return [], []


def get_parent_book_name(file_path: Path) -> str:
    name = file_path.parent.name
    return name if name else file_path.stem


def load_json_entries(file_path: Path) -> List[dict]:
    entries: List[dict] = []
    suffix = file_path.suffix.lower()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if suffix == ".json":
                raw = json.load(f)
                if isinstance(raw, list):
                    entries = [entry for entry in raw if isinstance(entry, dict)]
                elif isinstance(raw, dict):
                    entries = [raw]
            elif suffix == ".jsonl":
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        print(msg_debug(f"Skipping invalid JSONL row in {file_path.name}: {line[:40]}"))
                        continue
                    if isinstance(obj, dict):
                        entries.append(obj)
    except Exception as exc:
        print(msg_error(f"Failed to load {file_path}: {exc}"))
    return entries


async def process_text_files(
    pipeline: QAPipeline,
    text_files: List[Path],
    batch_size: int,
    start_index: int,
) -> None:
    if not text_files:
        return

    print(msg_success(f"Processing {len(text_files)} text/md file(s)."))
    output_dir = pipeline.output_dir

    parent_outputs: Dict[str, Path] = {}
    for file_path in text_files:
        parent_name = get_parent_book_name(file_path)
        parent_outputs[parent_name] = output_dir / f"{parent_name}.jsonl"

    source_files = text_files[start_index - 1 :] if start_index > 1 else text_files
    if start_index > 1:
        print(msg_info(f"Skipping first {start_index - 1} text file(s) for resume."))

    batch_texts: List[str] = []
    batch_sources: List[Path] = []
    batch_parents: List[str] = []
    for file_path in tqdm.tqdm(source_files, desc="Text files"):
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            print(msg_error(f"Failed to read {file_path}: {exc}"))
            continue
        if not text.strip():
            print(msg_debug(f"Skipping empty file: {file_path.name}"))
            continue
        batch_texts.append(text)
        batch_sources.append(file_path)
        batch_parents.append(get_parent_book_name(file_path))

    if not batch_texts:
        return

    async def on_result(item_id: int, result: Dict[str, Any]) -> None:
        source_path = batch_sources[item_id]
        parent_name = batch_parents[item_id]
        output_jsonl = parent_outputs[parent_name]
        if result.get("failed"):
            failure_path = output_jsonl.with_suffix(".failures.jsonl")
            result["source_files"] = [str(source_path.name)]
            result["id"] = str(uuid.uuid4())
            pipeline.append_failure_jsonl(failure_path, result)
            print(msg_error(f"Failed to create QA for: {source_path}. Saved failure to: {failure_path}"))
            return
        result["source_files"] = [str(source_path.name)]
        result["id"] = str(uuid.uuid4())
        pipeline.append_jsonl(output_jsonl, result)
        print(msg_info(f"Saved QA to: {output_jsonl}"))

    await pipeline.create_qa_batch_async(
        batch_texts,
        batch_size=batch_size,
        on_result=on_result,
    )


async def process_json_files(
    pipeline: QAPipeline,
    json_files: List[Path],
    target_key: str,
    batch_size: int,
    start_index: int,
) -> None:
    if not json_files:
        return

    print(msg_success(f"Processing {len(json_files)} JSON/JSONL file(s)."))

    output_path = Path(pipeline.settings.get("output_path", "./output")).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    for file_path in json_files:
        entries = load_json_entries(file_path)

        book_name = get_parent_book_name(file_path)
        cache_file = output_path / f"cache_{book_name}_{file_path.stem}.txt"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                processed_ids = set(line.strip() for line in f if line.strip())
                print(msg_info(f"Loaded {len(processed_ids)} processed IDs from cache for {file_path.name}."))
        else:
            processed_ids = set()

        print(msg_info(f"Before filtering, {len(entries)} entries in {file_path.name}."))
        entries = [entry for entry in entries if entry.get("id") not in processed_ids]
        print(msg_info(f"After filtering, {len(entries)} entries to process in {file_path.name}."))

        if not entries:
            print(msg_error(f"No entries found in {file_path}."))
            continue

        print(msg_info(f"Loaded {len(entries)} entries from {file_path.name}."))

        if start_index > 1:
            print(msg_info(f"Skipping first {start_index - 1} row(s) in {file_path.name} for resume."))
            entries = entries[start_index - 1 :]

        output_jsonl = output_path / f"{file_path.stem}.jsonl"
        batch_texts: List[str] = []
        ids: List[str] = []
        for entry in tqdm.tqdm(entries, desc=f"Entries / {file_path.name}"):
            value = entry.get(target_key)
            if not value or not isinstance(value, str):
                print(msg_debug(f"Entry missing target key {target_key} in {file_path.name}: {entry}"))
                continue
            batch_texts.append(value)
            ids.append(entry.get("id", str(uuid.uuid4())))

        if not batch_texts:
            continue

        async def on_result(item_id: int, result: Dict[str, Any]) -> None:
            entry_id = ids[item_id]
            if result.get("failed"):
                failure_path = output_jsonl.with_suffix(".failures.jsonl")
                result["source_files"] = [str(file_path.name)]
                result["id"] = entry_id
                pipeline.append_failure_jsonl(failure_path, result)
                print(msg_error(f"Failed to create QA for entry_id={entry_id}. Saved failure to: {failure_path}"))
                return
            result["source_files"] = [str(file_path.name)]
            result["id"] = entry_id
            pipeline.append_jsonl(output_jsonl, result)
            pipeline.add_cache(entry_id, f"{book_name}_{file_path.stem}")

        results = await pipeline.create_qa_batch_async(
            batch_texts,
            batch_size=batch_size,
            on_result=on_result,
        )
        print(msg_debug(f"Batch results count: {len(results)}"))

        print(msg_info(f"Saved QA to: {output_jsonl}"))


async def main(
    settings_path: str | None,
    source_path: str | None,
    target_key: str | None,
    start_index: int,
) -> None:
    if settings_path is None:
        print(msg_error("settings_path is required."), file=sys.stderr)
        sys.exit(1)
    settings = load_settings(Path(settings_path))

    if source_path is None:
        print(msg_error("source path is required."), file=sys.stderr)
        sys.exit(1)
    source = Path(source_path).expanduser().resolve()

    text_files, json_files = collect_source_files(source)
    if not text_files and not json_files:
        print(msg_error(f"No supported files were found for {source}."), file=sys.stderr)
        sys.exit(1)

    if json_files and not target_key:
        print(msg_error("target_key is required when processing JSON files."), file=sys.stderr)
        sys.exit(1)

    pipeline = QAPipeline(settings)
    batch_size = int(
        settings.get(
            "pipeline_batch_size",
            max(int(settings.get("batch_size", 1)), pipeline.max_in_flight * 4),
        )
    )
    print(
        msg_info(
            f"Async request concurrency max_in_flight={pipeline.max_in_flight}, "
            f"input_window_hint={batch_size}"
        )
    )
    try:
        await process_text_files(pipeline, text_files, batch_size, start_index)
        if target_key:
            await process_json_files(pipeline, json_files, target_key, batch_size, start_index)
    finally:
        await pipeline.aclose()


if __name__ == "__main__":
    print(msg_success("Imabari Q&A Creation Pipeline Started"))

    parser = argparse.ArgumentParser(description="Create Q&A from text, markdown, and json files (httpx async).")
    parser.add_argument(
        "-p",
        "--settings_path",
        nargs="?",
        default="./yamls/create_qa_settings.yaml",
        help="Path to the settings YAML file",
    )
    parser.add_argument(
        "-s",
        "--source",
        nargs="?",
        default=None,
        help="Path to a file or a directory containing sources",
    )
    parser.add_argument(
        "-t",
        "--target_key",
        type=str,
        default=None,
        help="Target key to extract from JSON/JSONL files",
    )
    parser.add_argument(
        "-i",
        "--start_index",
        type=int,
        default=0,
        help="Start index for resuming processing",
    )

    args = parser.parse_args()

    asyncio.run(
        main(
            settings_path=args.settings_path,
            source_path=args.source,
            target_key=args.target_key,
            start_index=args.start_index,
        )
    )

    print(msg_success("Imabari Q&A Creation Pipeline Completed"))
