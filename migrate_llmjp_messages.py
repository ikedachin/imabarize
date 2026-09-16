"""Rebuild LLM-jp messages only; preserve Qwen rows and all other values."""
import argparse
from pathlib import Path

from migrate_reasoning_effort_output import migrate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        count = migrate(args.file, args.output, messages_only=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'エラー: {exc}\n')
    print(f'完了: {count}件 → {args.output}（再開情報: {args.output}.resume.jsonl）')


if __name__ == '__main__':
    main()
