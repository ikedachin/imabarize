# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Remove a top-level key from every JSONL record, replacing the file atomically."""
import argparse
import json
import os
from pathlib import Path
import stat
import tempfile


def delete_key(path: Path, key: str) -> tuple[int, int]:
    path = path.expanduser().resolve(strict=True)
    temporary = None
    records = removed = 0
    try:
        with path.open("r", encoding="utf-8") as source:
            original = os.fstat(source.fileno())
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False,
            ) as output:
                temporary = Path(output.name)
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        output.write(line)
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError as exc:
                        raise ValueError(f"{line_number}行目: 不正なJSONです: {exc}") from exc
                    if not isinstance(record, dict):
                        raise ValueError(f"{line_number}行目: JSONオブジェクトではありません")
                    records += 1
                    if key in record:
                        del record[key]
                        removed += 1
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    else:
                        output.write(line)
                output.flush()
                os.fchmod(output.fileno(), stat.S_IMODE(original.st_mode))
                os.fsync(output.fileno())
        current = path.stat()
        if (current.st_ino, current.st_size, current.st_mtime_ns) != (
            original.st_ino, original.st_size, original.st_mtime_ns
        ):
            raise ValueError("処理中にファイルが変更されたため、置き換えを中止しました")
        if removed:
            os.replace(temporary, path)
        return records, removed
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="JSONLの各行から指定キーを削除し、元ファイルを更新します。")
    parser.add_argument("--file", required=True, type=Path, help="更新するJSONLファイル")
    parser.add_argument("--del-key", required=True, help="削除するトップレベルのキー")
    args = parser.parse_args()
    try:
        records, removed = delete_key(args.file, args.del_key)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"エラー: {exc}\n")
    print(f"完了: {records}件中{removed}件から {args.del_key!r} を削除しました。")


if __name__ == "__main__":
    main()
