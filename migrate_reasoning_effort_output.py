# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""One-off conversion of legacy Qwen/LLM-jp JSONL to the current output schema.

Uses only the Python standard library and local schema helpers; no inference or
tokenizer downloads. Keep the input .resume.jsonl beside the input JSONL when
the legacy export omits canonical IDs or tokenizer metadata.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import tempfile

from reasoning_effort.output_schema import compact_output, output_family, llmjp_messages, validate_llmjp_messages, template_kwargs


def migrate(source: Path, destination: Path, *, messages_only=False) -> int:
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
        metadata_by_digest = {}
        for entry in metadata.values():
            if entry.get('output_digest'):
                metadata_by_digest.setdefault(entry['output_digest'], []).append(entry)
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
                        if 'qa_id' not in row:
                            matches = metadata_by_digest.get(key, [])
                            if len(matches) != 1:
                                raise ValueError('Missing or ambiguous output resume metadata')
                            key = matches[0]['canonical_record_id']
                        if family is not None and family != detected:
                            raise ValueError('Qwenとllm-jpが混在しています')
                        family = detected
                        if key in seen:
                            raise ValueError('重複したqa_idです')
                        seen.add(key)
                        if detected == 'llm_jp_4':
                            row = dict(row, messages=llmjp_messages(row))
                            validate_llmjp_messages(row)
                        if messages_only:
                            # Validate without using the compacted result: every other
                            # key (including unknown legacy metadata) must survive.
                            compact_output(dict(row, qa_id=f"{key}:{detected}",
                                                chat_template_kwargs=template_kwargs(detected, row["reasoning_effort"])))
                            compact = row
                        else:
                            compact = compact_output(dict(row, qa_id=f"{key}:{detected}",
                                chat_template_kwargs=template_kwargs(detected, row["reasoning_effort"])))
                        info = row if 'target_tokenizer' in row else metadata.get(key, {})
                        if not info.get('target_tokenizer') or not info.get('target_tokenizer_revision'):
                            raise ValueError('Tokenizer情報がありません。新形式の場合は元の.resume.jsonlも必要です')
                        if row.get('canonical_record_id', key) != key:
                            raise ValueError('qa_idとcanonical_record_idが一致しません')
                        out.write(json.dumps(compact, ensure_ascii=False, allow_nan=False) + '\n')
                        meta.write(json.dumps({
                            'canonical_record_id': key,
                            'output_digest': output_family(compact)[0],
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
    parser = argparse.ArgumentParser(
        description='一時変換用: Qwen3.8／llm-jp-4の旧JSONLを新フォーマットへ変換します。元ファイルは変更しません。',
        epilog='モデルは自動判別します。ID・Tokenizer情報が省略された旧出力には、隣接する <入力>.resume.jsonl が必要です。'
               'LLM・Tokenizerは呼び出さず、thinking_tokensも保持します。',
    )
    parser.add_argument('--file', type=Path, required=True, help='変換元JSONL（1ファイルにつき1モデル）')
    parser.add_argument('--output', type=Path, required=True, help='未使用の出力JSONLパス。対応する.resume.jsonlも作成')
    args = parser.parse_args()
    try:
        count = migrate(args.file, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'エラー: {exc}\n')
    print(f'完了: {count}件 → {args.output}（再開情報: {args.output}.resume.jsonl）')


if __name__ == '__main__':
    main()
