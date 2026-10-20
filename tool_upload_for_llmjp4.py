"""Prepare and upload the standard Japanese LLM-jp 4 dataset as Parquet.

  .venv/bin/python tool_upload_standard_japanese_llmjp4.py --dry-run
  .venv/bin/python tool_upload_standard_japanese_llmjp4.py

HF_TOKEN is loaded from the project-root .env; existing environment values win.
Authentication also supports the cached Hugging Face login.
Requires a datasets release supporting List(Json), e.g. datasets 5.0.1.
For an isolated uv dependency override:
  uv run --with datasets==5.0.1 python tool_upload_standard_japanese_llmjp4.py --dry-run


# 通常版
uv run python tool_upload_standard_japanese_llmjp4.py \
  --input test_output/create_qa_v5_difficult_reasoning_effort/llmjp4.jsonl \
  --output-dir test_output/create_qa_v5_difficult_reasoning_effort/upload_parquet_llmjp4 \
  --repo-id ikedachin/imabari_wiki_qa_v5_difficult_reasoning_effort_llmjp4 \
  --allow-evaluated \
  --private

# 標準語版
uv run python tool_upload_standard_japanese_llmjp4.py \
  --input test_output/create_qa_v5_difficult_reasoning_effort_std_lang/llmjp4.jsonl \
  --output-dir test_output/create_qa_v5_difficult_reasoning_effort_std_lang/upload_parquet_llmjp4 \
  --repo-id ikedachin/std_lang_wiki_qa_v5_difficult_reasoning_effort_llmjp4 \
  --allow-evaluated \
  --private
"""

import argparse
import copy
import json
import os
import random
from datetime import datetime
from pathlib import Path

from datasets import Dataset, Features, Value
from dotenv import load_dotenv
from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "test_output/temporary_standard_japanese/llmjp4.jsonl"
DEFAULT_OUTPUT = ROOT / "test_output/temporary_standard_japanese/upload_parquet"
DEFAULT_REPO = "ikedachin/std_lang_wiki_qa_v4_validated_w_reasoning_effort_llmjp4"
FIELDS = (
    "qa_id", "question", "answer", "thinking", "reasoning_effort", "eval",
    "messages", "chat_template_kwargs", "thinking_tokens", "thinking_generator",
    "source_metadata",
)
REDUNDANT = {"source_qa_id": "qa_id", "id": "id", "chunk_index": "chunk_index"}


def reference_features() -> Features:
    """Pinned reference schema, including JSON messages and timestamp dates."""
    string = {"dtype": "string", "_type": "Value"}
    integer = {"dtype": "int64", "_type": "Value"}
    schema = {key: dict(string) for key in FIELDS}
    schema.update({
        "messages": {"feature": {"_type": "Json"}, "_type": "List"},
        "chat_template_kwargs": {
            "reasoning_effort": dict(string),
            "conversation_start_date": {"dtype": "timestamp[s]", "_type": "Value"},
        },
        "thinking_tokens": integer,
        "source_metadata": {"qa_id": dict(string), "id": dict(string), "chunk_index": integer},
    })
    try:
        return Features.from_dict(schema)
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError("List(Json) 対応の datasets が必要です（例: datasets==5.0.1）。") from exc


def read_records(path: Path, allow_evaluated: bool = False) -> list[dict]:
    records = []
    seen = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("レコードはobjectである必要があります")
            if set(row) - set(FIELDS) - set(REDUNDANT):
                raise ValueError("未知の項目があるため、情報の欠落を避けて停止します")
            record = {key: copy.deepcopy(row[key]) for key in FIELDS}
            if set(record["source_metadata"]) != {"qa_id", "id", "chunk_index"}:
                raise ValueError("source_metadata の項目が参照先と異なります")
            if set(record["chat_template_kwargs"]) != {"reasoning_effort", "conversation_start_date"}:
                raise ValueError("chat_template_kwargs の項目が参照先と異なります")
            # Generated datasets can keep the source ID only at the top level.
            # Preserve it in the reference schema before dropping redundant keys.
            if record["source_metadata"]["qa_id"] is None and row.get("source_qa_id") is not None:
                record["source_metadata"]["qa_id"] = row["source_qa_id"]
            for top, nested in REDUNDANT.items():
                if top in row and row[top] != record["source_metadata"][nested]:
                    raise ValueError(f"{top} と source_metadata.{nested} が一致しません")
            qa_id = record["qa_id"]
            if not isinstance(qa_id, str) or not qa_id or qa_id in seen:
                raise ValueError("qa_id が空、不正、または重複しています")
            for key in ("question", "answer", "thinking", "thinking_generator"):
                if not isinstance(record[key], str) or not record[key].strip():
                    raise ValueError(f"{key} は空でない文字列が必要です")
            if record["eval"] is not None and not (allow_evaluated and isinstance(record["eval"], str)):
                raise ValueError("標準語化後の未評価データを想定しているため eval は null が必要です")
            effort = record["reasoning_effort"]
            if effort not in {"low", "medium", "high"}:
                raise ValueError("未知の reasoning_effort です")
            if record["chat_template_kwargs"]["reasoning_effort"] != effort:
                raise ValueError("chat_template_kwargs の effort が一致しません")
            expected_messages = [
                {"role": "user", "content": record["question"]},
                {"role": "assistant", "content": record["answer"], "thinking": record["thinking"]},
            ]
            if record["messages"] != expected_messages:
                raise ValueError("messages と question/answer/thinking が一致しません")
            if type(record["thinking_tokens"]) is not int or record["thinking_tokens"] < 0:
                raise ValueError("thinking_tokens が不正です")
            kwargs = record["chat_template_kwargs"]
            date = kwargs["conversation_start_date"]
            kwargs["conversation_start_date"] = datetime.strptime(date, "%Y-%m-%d")
            if kwargs["conversation_start_date"].strftime("%Y-%m-%d") != date:
                raise ValueError("日付は YYYY-MM-DD 形式が必要です")
            seen.add(qa_id)
            records.append(record)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if len(records) < 2:
        raise ValueError("train/validation 分割には2件以上必要です")
    return records


def prepare(input_path: Path, output: Path,
            ratio: float = 0.1, seed: int = 42, allow_evaluated: bool = False) -> dict[str, int]:
    if not 0 < ratio < 1:
        raise ValueError("validation-ratio は0より大きく1未満が必要です")
    records = read_records(input_path, allow_evaluated=allow_evaluated)
    features = reference_features()
    random.Random(seed).shuffle(records)
    count = max(1, min(int(len(records) * ratio), len(records) - 1))
    raw_splits = {"train": records[count:], "validation": records[:count]}
    splits = {name: Dataset.from_list(rows, features=features) for name, rows in raw_splits.items()}
    (output / "data").mkdir(parents=True, exist_ok=True)
    for name, dataset in splits.items():
        path = output / "data" / f"{name}-00000-of-00001.parquet"
        dataset.to_parquet(path)
        # Read without a datasets cache or network; check every value after encoding.
        import pyarrow.parquet as pq
        restored = Dataset(pq.read_table(path))
        # Parquet stores second-resolution timestamps as milliseconds, as in
        # the reference repository. Values must still match exactly.
        parquet_features = copy.deepcopy(features)
        parquet_features["chat_template_kwargs"]["conversation_start_date"] = Value("timestamp[ms]")
        if restored.features != parquet_features or list(restored) != raw_splits[name]:
            raise ValueError(f"{path}: Parquet再読込で型または値が変化しました")
    return {name: len(dataset) for name, dataset in splits.items()}


def upload(output: Path, repo_id: str, private: bool = False) -> str:
    load_dotenv(ROOT / ".env", override=False)
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)  # Never print credentials.
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    result = api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=output,
        allow_patterns=["data/train-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet"],
        commit_message="Upload standard Japanese LLM-jp 4 dataset with reference Parquet schema",
    )
    return result.commit_url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-evaluated", action="store_true", help="文字列の eval を保持して取り込む（未評価の null も許可）")
    parser.add_argument("--private", action="store_true", help="新規作成時のみ非公開にする。既存repoの公開設定は変更しない")
    parser.add_argument("--dry-run", action="store_true", help="Parquetを生成・検証し、アップロードせず終了")
    args = parser.parse_args()
    counts = prepare(args.input, args.output_dir, args.validation_ratio, args.seed, args.allow_evaluated)
    print(f"Prepared: {counts}; output={args.output_dir.resolve()}")
    print(f"Destination: https://huggingface.co/datasets/{args.repo_id}")
    if args.dry_run:
        print("Dry run completed. No network requests or uploads were made.")
    else:
        print(f"Uploaded: {upload(args.output_dir, args.repo_id, args.private)}")


if __name__ == "__main__":
    main()
