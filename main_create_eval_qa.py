import argparse
import asyncio
import random
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from commons.util_settings import load_settings
from commons.utils_msg import msg_debug, msg_error, msg_info, msg_success
from main_create_imabari_qa import (
    collect_source_files,
    entry_cache_key,
    get_parent_book_name,
    is_entry_processed,
    load_json_entries,
    load_processed_cache_keys,
)
from pipelines.create_qa_model import QAPipeline


def parse_sample_size(value: Any) -> Optional[int]:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


def select_sampled_items(
    items: List[Dict[str, Any]],
    sample_size: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    if sample_size is None or len(items) <= sample_size:
        return list(items)
    return random.Random(seed).sample(items, sample_size)


async def process_eval_text_files(
    pipeline: QAPipeline,
    text_files: List[Path],
    batch_size: int,
    start_index: int,
    sample_size: Optional[int],
    seed: int,
) -> int:
    if not text_files:
        return 0

    output_dir = pipeline.output_dir
    source_files = text_files[start_index - 1 :] if start_index > 1 else text_files
    if start_index > 1:
        print(msg_info(f"Skipping first {start_index - 1} text file(s) for resume."))

    candidates: List[Dict[str, Any]] = []
    for file_path in source_files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            print(msg_error(f"Failed to read {file_path}: {exc}"))
            continue
        if not text.strip():
            print(msg_debug(f"Skipping empty file: {file_path.name}"))
            continue
        parent_name = get_parent_book_name(file_path)
        candidates.append(
            {
                "text": text,
                "source_path": file_path,
                "output_jsonl": output_dir / f"{parent_name}.jsonl",
            }
        )

    selected = select_sampled_items(candidates, sample_size, seed)
    print(
        msg_success(
            f"Processing evaluation text/md file(s): candidates={len(candidates)}, selected={len(selected)}."
        )
    )
    if not selected:
        return 0

    async def on_result(item_id: int, result: Dict[str, Any]) -> None:
        candidate = selected[item_id]
        source_path = candidate["source_path"]
        output_jsonl = candidate["output_jsonl"]
        if result.get("failed"):
            failure_path = output_jsonl.with_suffix(".failures.jsonl")
            result["source_files"] = [str(source_path.name)]
            result["id"] = str(uuid.uuid4())
            result["chunk_index"] = None
            pipeline.append_failure_jsonl(failure_path, result)
            print(msg_error(f"Failed to create eval QA for: {source_path}. Saved failure to: {failure_path}"))
            return
        result["source_files"] = [str(source_path.name)]
        result["id"] = str(uuid.uuid4())
        result["chunk_index"] = None
        pipeline.append_jsonl(output_jsonl, result)
        print(msg_info(f"Saved eval QA to: {output_jsonl}"))

    await pipeline.create_qa_batch_async(
        [str(candidate["text"]) for candidate in selected],
        batch_size=batch_size,
        on_result=on_result,
    )
    return len(selected)


async def process_eval_json_files(
    pipeline: QAPipeline,
    json_files: List[Path],
    target_key: str,
    batch_size: int,
    start_index: int,
    sample_size: Optional[int],
    seed: int,
) -> int:
    if not json_files:
        return 0

    output_path = Path(pipeline.settings.get("output_path", "./output")).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    candidates: List[Dict[str, Any]] = []

    for file_path in json_files:
        entries = load_json_entries(file_path)
        for row_index, entry in enumerate(entries):
            entry["_qa_entry_id"] = str(entry.get("id") or f"{file_path.stem}:{row_index}")

        book_name = get_parent_book_name(file_path)
        cache_name = f"{book_name}_{file_path.stem}"
        cache_file = output_path / f"cache_{cache_name}.txt"
        processed_keys = load_processed_cache_keys(cache_file)
        print(msg_info(f"Loaded {len(processed_keys)} processed cache keys from cache for {file_path.name}."))

        pending_entries = [
            entry
            for entry in entries
            if not is_entry_processed(
                str(entry["_qa_entry_id"]),
                entry.get("chunk_index"),
                processed_keys,
            )
        ]
        if start_index > 1:
            print(msg_info(f"Skipping first {start_index - 1} row(s) in {file_path.name} for resume."))
            pending_entries = pending_entries[start_index - 1 :]

        output_jsonl = output_path / f"{file_path.stem}.jsonl"
        for entry in pending_entries:
            value = entry.get(target_key)
            if not value or not isinstance(value, str):
                print(msg_debug(f"Entry missing target key {target_key} in {file_path.name}: {entry}"))
                continue
            entry_id = str(entry["_qa_entry_id"])
            chunk_index = entry.get("chunk_index")
            candidates.append(
                {
                    "text": value,
                    "source_file": file_path.name,
                    "output_jsonl": output_jsonl,
                    "entry_id": entry_id,
                    "chunk_index": chunk_index,
                    "cache_key": entry_cache_key(entry_id, chunk_index),
                    "cache_name": cache_name,
                }
            )

    selected = select_sampled_items(candidates, sample_size, seed)
    print(
        msg_success(
            f"Processing evaluation JSON/JSONL entries: candidates={len(candidates)}, selected={len(selected)}."
        )
    )
    if not selected:
        return 0

    async def on_result(item_id: int, result: Dict[str, Any]) -> None:
        candidate = selected[item_id]
        output_jsonl = candidate["output_jsonl"]
        entry_id = str(candidate["entry_id"])
        chunk_index = candidate["chunk_index"]
        if result.get("failed"):
            failure_path = output_jsonl.with_suffix(".failures.jsonl")
            result["source_files"] = [str(candidate["source_file"])]
            result["id"] = entry_id
            result["chunk_index"] = chunk_index
            pipeline.append_failure_jsonl(failure_path, result)
            print(
                msg_error(
                    f"Failed to create eval QA for entry_id={entry_id} chunk_index={chunk_index}. "
                    f"Saved failure to: {failure_path}"
                )
            )
            return
        result["source_files"] = [str(candidate["source_file"])]
        result["id"] = entry_id
        result["chunk_index"] = chunk_index
        pipeline.append_jsonl(output_jsonl, result)
        pipeline.add_cache(str(candidate["cache_key"]), str(candidate["cache_name"]))

    results = await pipeline.create_qa_batch_async(
        [str(candidate["text"]) for candidate in selected],
        batch_size=batch_size,
        on_result=on_result,
    )
    print(msg_debug(f"Evaluation QA batch results count: {len(results)}"))
    return len(selected)


async def main(
    settings_path: str | None,
    source_path: str | None,
    target_key: str | None,
    start_index: int,
    sample_size_override: Optional[int] = None,
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

    sample_size = sample_size_override
    if sample_size is None:
        sample_size = parse_sample_size(settings.get("sample_size"))
    seed = int(settings.get("seed", 42))

    pipeline = QAPipeline(settings)
    batch_size = int(
        settings.get(
            "pipeline_batch_size",
            max(int(settings.get("batch_size", 1)), pipeline.max_in_flight * 4),
        )
    )
    sample_label = sample_size if sample_size is not None else "all"
    print(
        msg_info(
            f"Async request concurrency max_in_flight={pipeline.max_in_flight}, "
            f"input_window_hint={batch_size}, sample_size={sample_label}, seed={seed}"
        )
    )
    try:
        selected_text_count = await process_eval_text_files(
            pipeline,
            text_files,
            batch_size,
            start_index,
            sample_size,
            seed,
        )
        if target_key:
            json_sample_size = sample_size
            if sample_size is not None:
                remaining = sample_size - selected_text_count
                if remaining <= 0:
                    print(msg_info("Skipping evaluation JSON/JSONL entries because sample_size is already reached."))
                    return
                json_sample_size = remaining
            await process_eval_json_files(
                pipeline,
                json_files,
                target_key,
                batch_size,
                start_index,
                json_sample_size,
                seed,
            )
    finally:
        await pipeline.aclose()


if __name__ == "__main__":
    print(msg_success("Evaluation Q&A Creation Pipeline Started"))

    parser = argparse.ArgumentParser(
        description="Create evaluation Q&A from text, markdown, and json files (httpx async pipeline pool)."
    )
    parser.add_argument(
        "-p",
        "--settings_path",
        nargs="?",
        default="./yamls/create_eval_qa_settings_format.yaml",
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
    parser.add_argument(
        "-n",
        "--sample_size",
        type=int,
        default=None,
        help="Maximum number of evaluation Q&A records to create. Defaults to YAML sample_size or all.",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            settings_path=args.settings_path,
            source_path=args.source,
            target_key=args.target_key,
            start_index=args.start_index,
            sample_size_override=parse_sample_size(args.sample_size),
        )
    )

    print(msg_success("Evaluation Q&A Creation Pipeline Completed"))
