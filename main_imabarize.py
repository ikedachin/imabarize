import argparse
import json
import sys
from pathlib import Path

import tqdm
import yaml

from pipelines.imabarize_pipeline import SanitizePipeline
from commons.util_dataloader import ListDataLoader
from commons.utils_msg import msg_info, msg_debug, msg_error, msg_success


"""
出力形式：jsonl
各行が1つのJSONオブジェクトである必要があります。
データ
{
    "book": "book_name", # ファイル名と同じ
    "page": 1,
    "text": "This is the content of the book.",
    "sanitized_text": "This is the sanitized content of the book.",
    "eval_score": 0.9, # llm as a judge
    "similarity": 0.8, # tfidf
}
"""


def get_files_to_sanitize(source_files: list[Path], settings: dict) -> set[str]:
    # 実行済みのファイルをスキップするためのプログラム
    print(msg_debug(f"Function start: {get_files_to_sanitize.__name__}"))
    output_path = Path(settings["output_path"]).expanduser().resolve()
    if not output_path.exists():
        output_path.mkdir(parents=True)

    sanitized_files = sorted(output_path.glob("**/*.jsonl"))
    loaded_files = load_files(source_files, settings)  # keys = book, page, <target_key>

    sanitized_info = {}
    for sanitized_file in sanitized_files:
        with open(sanitized_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data['book'] not in sanitized_info:
                    sanitized_info[data['book']] = [data['page']]
                else:
                    sanitized_info[data['book']].append(data['page'])
    print(msg_debug(f"sanitized_info: {sanitized_info}"))

    output_files = []
    for loaded_file in loaded_files:
        book = loaded_file['book']
        page = loaded_file['page']
        if book in sanitized_info and page in sanitized_info[book]:
            print(msg_debug(f"Sanitized: {loaded_file}"))
        else:
            output_files.append(loaded_file)
    print(msg_debug(f"Function end: {get_files_to_sanitize.__name__}"))
    return output_files


def _load_json_entries(file_path: Path) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    entries = raw if isinstance(raw, list) else [raw]
    return [entry for entry in entries if isinstance(entry, dict)]


def _load_jsonl_entries(file_path: Path) -> list[dict]:
    entries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
    return entries


def load_files(source_files: list[Path], settings: dict | None = None) -> list[dict]:
    """
    Load entries from .json / .jsonl files and return list of records.
    If settings contains 'target_key', use it to extract the text field and
    exclude that key from extras. Otherwise fall back to 'text' or 'content'.
    """
    loaded_files = []
    target_key = None
    if settings and isinstance(settings, dict):
        target_key = settings.get("target_key")

    for file_path in sorted(source_files):
        print(msg_debug(file_path))
        # support if source_files contains dict entries already
        if isinstance(file_path, dict):
            entry_items = [file_path]
            file_path_for_name = None
        else:
            suffix = file_path.suffix.lower()
            if suffix not in {".json", ".jsonl"}:
                continue
            if suffix == ".json":
                entry_items = _load_json_entries(file_path)
            else:
                entry_items = _load_jsonl_entries(file_path)
            file_path_for_name = file_path

        for entry in entry_items:
            if not isinstance(entry, dict):
                continue
            # choose text based on target_key or fallback
            if target_key:
                text = entry.get(target_key)
            else:
                text = entry.get("text") or entry.get("content")
            if not text:
                continue
            book = entry.get("book") or (file_path_for_name.stem if file_path_for_name else None)
            page = entry.get("page")

            # build exclude keys set depending on target_key
            exclude_keys = {"book", "page"}
            if target_key:
                exclude_keys.add(target_key)
            else:
                exclude_keys.update({"text", "content"})

            extras = {k: v for k, v in entry.items() if k not in exclude_keys}

            record = {
                "book": book,
                "page": page,
                (target_key if target_key else "text"): text,
                **extras,
            }
            loaded_files.append(record)
    return loaded_files




def main(args) -> None:
    # args contain settings_path source, target_keys, start_index

    # ======================================================
    # 条件設定
    # ======================================================

    # 設定値の取得
    if args.settings_path is None:
        print(msg_debug("settings is required."))
        return
    else:
        settings_path = Path(args.settings_path).expanduser().resolve()

        with open(settings_path, "r", encoding="utf-8") as f:
            settings = yaml.safe_load(f)
            print(msg_debug(f"Settings loaded: \n{settings}"))
    print(msg_debug(f"Loading settings from: {settings_path}"))

    # 引数をすべて設定に追加
    if args.extensions:
        settings["extensions"] = ["." + ext.strip() if not ext.startswith(".") else ext.strip() for ext in args.extensions.split(",")]
        print(msg_debug(f"Extensions: {settings['extensions']}"))

    # ターゲットキーの取得
    # CLI で指定があればそれを優先、なければ設定ファイルの値を尊重し、どちらにも無ければ 'text' をデフォルトにする
    if args.target_key is not None:
        settings['target_key'] = args.target_key
        print(msg_debug(f"Target key (from CLI): {args.target_key}"))
    else:
        if not settings.get('target_key'):
            settings['target_key'] = 'text'
            print(msg_debug("Target key not provided; defaulting to 'text'."))
        else:
            print(msg_debug(f"Target key (from settings): {settings['target_key']}"))

    # ソースパスの取得
    if args.source is None:
        print(msg_debug("source is required."))
        return
    else:
        # for file in Path(args.source).expanduser().resolve().glob("**/*.txt"):
        #     print(file)
        source_path = Path(args.source).expanduser().resolve()
        settings["source_path"] = source_path
        print(msg_debug(f"Source path: {source_path}"))
        if source_path.is_dir():
            source_files = []
            for file in source_path.glob("**/*"):
                if file.is_file():
                    if args.extensions:
                        if file.suffix in settings['extensions']:
                            source_files.append(file)
                        else:
                            print(msg_debug(f"File excluded by extension: {file}"))
                    else:
                        source_files.append(file)
            source_files = sorted(source_files)
        else:
            source_files = [source_path]


    print(msg_debug(f"Source files Nums: {len(source_files)}"))

    # アウトプットファイルから実行されていないファイルだけにする
    source_files = get_files_to_sanitize(source_files, settings)


    # ======================================================
    # パイプラインの初期化と実行
    # ======================================================

    pipeline = SanitizePipeline(settings)

    dataloader = ListDataLoader(source_files, settings)

    for data, start_index, end_index in dataloader:

        # print(msg_debug(f"batched files: {data=}, {start_index=}, {end_index=}"))

        results = pipeline.imabarize_batch(
            batched_data=data,
        )
        if not results:
            continue
        for result in results:
            pipeline.save_results(result)



if __name__ == "__main__":
    print(msg_success("Imabarize Pipeline Started"))

    parser = argparse.ArgumentParser(description="Imabarize datas.")
    parser.add_argument(
        "-s",
        "--source",
        nargs="?",
        default=None,
        help="Path to a file or a directory include .md .json, .jsonl, .txt",
    )
    parser.add_argument(
        "-p",
        "--settings_path",
        nargs="?",
        default="./yamls/imabari_settings.yaml",
        help="Path to the settings YAML file",
    )
    parser.add_argument(
        "-t",
        "--target_key",
        type=str,
        default=None,
        help="target key to extract from JSONL files",
    )
    parser.add_argument(
        "-i",
        "--start_index",
        type=int,
        default=0,
        help="Start index for resuming processing",
    )
    parser.add_argument(
        "-e",
        "--extensions",
        type=str,
        default=None,
        help="File extensions to process (e.g. .md, .json, .jsonl, .txt)",
    )
    
    args = parser.parse_args()

    main(args)

    print(msg_success("Sanitization Pipeline Completed"))
