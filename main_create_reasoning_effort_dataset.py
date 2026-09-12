"""CLI for the independent dual-target reasoning effort dataset generator."""
import argparse
import asyncio
import json
from pathlib import Path
from typing import Iterator

import yaml

from commons.utils_msg import msg_info, msg_success
from pipelines.create_reasoning_effort_dataset import ReasoningEffortDatasetPipeline


def load_source(settings: dict) -> Iterator[dict]:
    source = settings["source"]
    if source["type"] == "huggingface":
        from datasets import load_dataset
        yield from load_dataset(source["dataset_name"], name=source.get("config"),
                                split=source.get("split", "train"), streaming=True,
                                revision=source.get("revision", "main"))
    elif source["type"] == "local":
        with Path(source["path"]).expanduser().open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A malformed row is an input failure, not the end of the dataset.
                    yield {"source_line": number, "input_error": "invalid_json"}
    else:
        raise ValueError("source.type must be huggingface or local")


async def main(settings_path: str) -> dict:
    # Avoid the existing helper's full settings dump (API configuration may be sensitive).
    with Path(settings_path).open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    pipeline = ReasoningEffortDatasetPipeline(settings)
    try:
        summary = await pipeline.run(load_source(settings))
        print(msg_info(json.dumps(summary, ensure_ascii=False, indent=2)))
        return summary
    finally:
        await pipeline.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate shared Imabari thinking and two SFT JSONL datasets.")
    parser.add_argument("-p", "--settings_path", default="./yamls/create_reasoning_effort_dataset.yaml")
    args = parser.parse_args()
    print(msg_success("Reasoning Effort Dataset Generator Started"))
    result = asyncio.run(main(args.settings_path))
    if any(value for key, value in result.items() if key.endswith("_terminal_failures")):
        raise SystemExit(1)
    print(msg_success("Reasoning Effort Dataset Generator Completed"))
