# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""One-off directory migration. Dry-run by default; no inference or tokenizer downloads.

uv run python migrate_reasoning_effort_directory.py
uv run python migrate_reasoning_effort_directory.py --apply

Stop dataset generation before running. Existing destination records take precedence.
Cache identities are preserved, not relabeled for changed prompts/settings.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

from migrate_reasoning_effort_output import migrate
from reasoning_effort.output_schema import output_family


OUTPUTS = ('qwen38.jsonl', 'llmjp4.jsonl')
CACHE = '.generation_cache.jsonl'
FAILURES = 'failures.jsonl'
FILES = (*OUTPUTS, *(name + '.resume.jsonl' for name in OUTPUTS), CACHE, FAILURES)


def read_rows(path):
    if not path.exists():
        return
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('JSON object required')
                yield row
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{path}:{number}: {exc}') from exc


def write_rows(path, rows):
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def build(source, destination, stage):
    """Validate and prepare all files before publishing any of them."""
    summary = {}
    cache = {}
    for directory in (source, destination):
        for row in read_rows(directory / CACHE):
            row = dict(row)
            key = row.get('canonical_record_id')
            if not key or row.get('journal_key', key) != key:
                raise ValueError(f'{directory / CACHE}: inconsistent cache identity')
            for field in ('source_qa_id', 'question', 'answer', 'thinking',
                          'canonical_reasoning_effort', 'canonical_thinking_tokens',
                          'canonical_tokenizer', 'canonical_tokenizer_revision'):
                if row.get(field) is None:
                    raise ValueError(f'{directory / CACHE}: missing {field}')
            row.pop('generation_context', None)
            cache[key] = row
    write_rows(stage / CACHE, cache.values())
    summary[CACHE] = len(cache)

    for name in OUTPUTS:
        outputs, metadata = {}, {}
        expected_family = 'qwen3_8' if name == 'qwen38.jsonl' else 'llm_jp_4'
        for index, directory in enumerate((source, destination)):
            path = directory / name
            if not path.exists() or path.stat().st_size == 0:
                continue
            converted = stage / f'{index}-{name}'
            migrate(path, converted)
            for row in read_rows(converted):
                key, family = output_family(row)
                if family != expected_family:
                    raise ValueError(f'{path}: unexpected model family {family}')
                outputs[key] = row
            for row in read_rows(Path(str(converted) + '.resume.jsonl')):
                metadata[row['canonical_record_id']] = row
        for key, row in outputs.items():
            canonical = cache.get(key)
            if canonical is None:
                raise ValueError(f'{name}: missing canonical cache: {key}')
            for field in ('source_qa_id', 'question', 'answer', 'thinking'):
                if row[field] != canonical[field]:
                    raise ValueError(f'{name}: cache/output conflict: {key}: {field}')
            effort = canonical['canonical_reasoning_effort']
            expected = 'xhigh' if expected_family == 'qwen3_8' and effort == 'high' else effort
            if row['reasoning_effort'] != expected:
                raise ValueError(f'{name}: cache/output effort conflict: {key}')
        write_rows(stage / name, outputs.values())
        write_rows(stage / (name + '.resume.jsonl'), (metadata[key] for key in outputs))
        summary[name] = len(outputs)
        summary[name + '.resume.jsonl'] = len(outputs)

    # Keep historical failures, including subsequently successful attempts.
    # Exact duplicates are removed, making repeated migrations idempotent.
    failures = {}
    for directory in (source, destination):
        for row in read_rows(directory / FAILURES):
            failures[json.dumps(row, sort_keys=True, ensure_ascii=False)] = row
    write_rows(stage / FAILURES, failures.values())
    summary[FAILURES] = len(failures)
    return summary


def migrate_directory(source, destination, *, apply=False):
    source = Path(source).expanduser().resolve(strict=True)
    destination = Path(destination).expanduser().resolve()
    if not source.is_dir() or source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('入力と出力は、互いに包含しない別ディレクトリにしてください')
    for name in (*OUTPUTS, CACHE):
        if not (source / name).is_file():
            raise ValueError(f'入力ファイルがありません: {source / name}')
    destination.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        # Same lock paths/order as the pipeline. Never rename a locked directory.
        for path in sorted({directory / name for directory in (source, destination) for name in FILES}):
            lock = stack.enter_context(Path(str(path) + '.lock').open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(f'生成または移行処理が実行中です。停止後に再実行してください: {path}') from exc
        stage = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='.migration-', dir=destination)))
        # The single-file converter takes its own input locks. Use snapshots so
        # directory locks remain held throughout conversion and publication.
        snapshots = []
        for index, directory in enumerate((source, destination)):
            snapshot = stage / f'input-{index}'
            snapshot.mkdir()
            for name in FILES:
                if (directory / name).exists():
                    shutil.copy2(directory / name, snapshot / name)
            snapshots.append(snapshot)
        summary = build(*snapshots, stage)
        if not apply:
            return {'mode': 'dry-run', 'records': summary}
        backup = Path(tempfile.mkdtemp(prefix='migration-backup-' + datetime.now().strftime('%Y%m%d-%H%M%S-'), dir=destination))
        existing = set()
        for name in FILES:
            if (destination / name).exists():
                shutil.copy2(destination / name, backup / name)
                existing.add(name)
        # Retain prepared files too, allowing recovery after a process/machine crash.
        prepared = backup / 'prepared'
        prepared.mkdir()
        for name in FILES:
            shutil.copy2(stage / name, prepared / name)
        (backup / 'manifest.json').write_text(json.dumps({
            'source': str(source), 'destination': str(destination),
            'previous_files': sorted(existing), 'records': summary,
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        published = []
        try:
            for name in FILES:
                os.replace(stage / name, destination / name)
                published.append(name)
        except BaseException:
            for name in reversed(published):
                if name in existing:
                    shutil.copy2(backup / name, destination / name)
                else:
                    (destination / name).unlink()
            raise
        return {'mode': 'applied', 'records': summary, 'backup': str(backup)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('test_output/reasoning_effort_with_context_old'))
    parser.add_argument('--destination', type=Path, default=Path('test_output/reasoning_effort_with_context'))
    parser.add_argument('--apply', action='store_true', help='検証後、バックアップを作成して反映')
    args = parser.parse_args()
    try:
        result = migrate_directory(args.source, args.destination, apply=args.apply)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'エラー: {exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
