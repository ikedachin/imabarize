# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Convert a completed reasoning export into a NEW compact JSONL and resume sidecar."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import tempfile

from reasoning_effort.output_schema import compact_output, output_family


def migrate(source: Path, destination: Path) -> int:
    source = source.expanduser().resolve(strict=True)
    destination = destination.expanduser().resolve()
    sidecar = Path(str(destination) + '.resume.jsonl')
    source_sidecar = Path(str(source) + '.resume.jsonl')
    if source in (destination, sidecar) or source_sidecar in (destination, sidecar):
        raise ValueError('入力と出力は別のパスにしてください')
    if destination.exists() or sidecar.exists():
        raise ValueError('出力またはresumeファイルが存在します。新しい出力パスを指定してください')
    destination.parent.mkdir(parents=True, exist_ok=True)
    locks, temporary, published = [], [], []
    try:
        for path in sorted({source, source_sidecar, destination, sidecar}):
            lock = Path(str(path) + '.lock').open('a')
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = {}
        if source_sidecar.exists():
            for line in source_sidecar.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                metadata[row['canonical_record_id']] = row
        count, family, seen = 0, None, set()
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=destination.parent,
                                         delete=False) as out, tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=destination.parent, delete=False) as meta:
            temporary.extend([Path(out.name), Path(meta.name)])
            with source.open(encoding='utf-8') as stream:
                before = os.fstat(stream.fileno())
                for number, line in enumerate(stream, 1):
                    try:
                        row = json.loads(line)
                        key, detected = output_family(row)
                        if family is not None and family != detected:
                            raise ValueError('Qwenとllm-jpが混在しています')
                        family = detected
                        if key in seen:
                            raise ValueError('重複したqa_idです')
                        seen.add(key)
                        # Only legacy migration reads the obsolete adapter field.
                        if detected == 'llm_jp_4' and any(
                            isinstance(m, dict) and ('channel' in m or isinstance(m.get('content'), list))
                            for m in row.get('messages', [])
                        ):
                            row = dict(row, messages=row.get('template_messages'))
                        compact = compact_output(row)
                        info = row if 'target_tokenizer' in row else metadata.get(key, {})
                        if not info.get('target_tokenizer') or not info.get('target_tokenizer_revision'):
                            raise ValueError('Tokenizer情報がありません。新形式の場合は元の.resume.jsonlも必要です')
                        if row.get('canonical_record_id', key) != key:
                            raise ValueError('qa_idとcanonical_record_idが一致しません')
                        out.write(json.dumps(compact, ensure_ascii=False, allow_nan=False) + '\n')
                        meta.write(json.dumps({
                            'canonical_record_id': key,
                            'target_tokenizer': info['target_tokenizer'],
                            'target_tokenizer_revision': info['target_tokenizer_revision'],
                        }, ensure_ascii=False, allow_nan=False) + '\n')
                        count += 1
                    except (ValueError, TypeError, KeyError, AttributeError) as exc:
                        raise ValueError(f'{number}行目: {exc}') from exc
                after = source.stat()
                if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                    raise ValueError('入力ファイルが処理中に変更されました')
            if not count:
                raise ValueError('入力ファイルが空です')
            for stream in (out, meta):
                stream.flush()
                os.fsync(stream.fileno())
        # Hard links publish complete files without overwriting an existing path.
        for temp, path in ((temporary[1], sidecar), (temporary[0], destination)):
            os.link(temp, path)
            published.append(path)
        return count
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
        for lock in locks:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description='既存Reasoning Effort JSONLを新しいファイルへ軽量化します。元ファイルは変更しません。')
    parser.add_argument('--file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        count = migrate(args.file, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'エラー: {exc}\n')
    print(f'完了: {count}件 → {args.output}（再開情報: {args.output}.resume.jsonl）')


if __name__ == '__main__':
    main()
