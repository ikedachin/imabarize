"""Copy generated answers to a new directory and recover their final channels.

Stop the generation process before running this tool. The original directory is
never modified; rows that cannot be safely parsed are left pending for retry.
"""

import argparse
import json
import shutil
from pathlib import Path

from pipelines.evaluate_llm_jp import EvaluationError, append_jsonl, extract_final_answer, read_jsonl


def recover(source: Path, destination: Path) -> tuple[int, int]:
    source, destination = source.expanduser().resolve(), destination.expanduser().resolve()
    if source == destination or destination.exists():
        raise EvaluationError("Destination must be a new directory different from the source")
    manifest, answers, results = source / "manifest.json", source / "answers.jsonl", source / "all.jsonl"
    if not manifest.is_file() or not answers.is_file():
        raise EvaluationError("Source must contain manifest.json and answers.jsonl")
    if results.exists() and results.stat().st_size:
        raise EvaluationError("Source contains judge results; use a separate review rather than discarding them")
    before = answers.stat()
    rows = read_jsonl(answers)
    after = answers.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise EvaluationError("Source answers changed while reading; stop generation and retry")
    clean, rejected = [], []
    for row in rows.values():
        try:
            answer, channel = extract_final_answer(row["answer"], None)
            clean.append({**row, "answer": answer, "observed_final_channel": channel or row.get("observed_final_channel")})
        except (EvaluationError, KeyError, TypeError) as exc:
            rejected.append({"record": row, "error": str(exc)})
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest, destination / "manifest.json")
    failures = source / "failures.jsonl"
    if failures.exists():
        shutil.copyfile(failures, destination / "failures.jsonl")
    for row in clean:
        append_jsonl(destination / "answers.jsonl", row)
    for row in rejected:
        append_jsonl(destination / "rejected_answers.jsonl", row)
    return len(clean), len(rejected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Existing generation output directory")
    parser.add_argument("--output", required=True, help="New output directory")
    args = parser.parse_args()
    try:
        saved, rejected = recover(Path(args.source), Path(args.output))
        print(json.dumps({"recovered": saved, "rejected": rejected, "output": str(Path(args.output).expanduser().resolve())}, ensure_ascii=False, indent=2))
        return 0
    except (EvaluationError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Recovery failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
