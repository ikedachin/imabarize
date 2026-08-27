import argparse
import asyncio
import sys
from pathlib import Path

from commons.util_settings import load_settings
from commons.utils_msg import msg_error, msg_info, msg_success
from pipelines.judge_eval_qa import JudgeEvalQAPipeline


async def main(
    settings_path: str | None,
    eval_qa_path: str | None,
    candidate_answer_path: str | None,
) -> None:
    if settings_path is None:
        print(msg_error("settings_path is required."), file=sys.stderr)
        sys.exit(1)

    settings = load_settings(Path(settings_path))
    configured_eval_qa_path = eval_qa_path or settings.get("eval_qa_path")
    configured_candidate_answer_path = candidate_answer_path or settings.get("candidate_answer_path")
    if not configured_eval_qa_path:
        print(msg_error("eval_qa_path is required."), file=sys.stderr)
        sys.exit(1)
    if not configured_candidate_answer_path:
        print(msg_error("candidate_answer_path is required."), file=sys.stderr)
        sys.exit(1)

    pipeline = JudgeEvalQAPipeline(settings)
    print(
        msg_info(
            f"Async request concurrency max_in_flight={pipeline.max_in_flight}, "
            f"input_window_hint={pipeline.pipeline_batch_size}"
        )
    )
    try:
        stats = await pipeline.judge_dataset(
            Path(configured_eval_qa_path).expanduser().resolve(),
            Path(configured_candidate_answer_path).expanduser().resolve(),
        )
        print(msg_success(f"Judge results saved to: {pipeline.output_dir}"))
        print(msg_info(f"Stats: {stats}"))
    finally:
        await pipeline.aclose()


if __name__ == "__main__":
    print(msg_success("Evaluation Q&A Judge Pipeline Started"))

    parser = argparse.ArgumentParser(
        description="Judge candidate answers against evaluation Q&A JSONL using an external LLM."
    )
    parser.add_argument(
        "-p",
        "--settings_path",
        nargs="?",
        default="./yamls/judge_eval_qa_settings_format.yaml",
        help="Path to the settings YAML file",
    )
    parser.add_argument(
        "-q",
        "--eval_qa",
        nargs="?",
        default=None,
        help="Path to evaluation Q&A JSON/JSONL. Defaults to settings eval_qa_path.",
    )
    parser.add_argument(
        "-a",
        "--candidate_answers",
        nargs="?",
        default=None,
        help="Path to candidate answer JSON/JSONL. Defaults to settings candidate_answer_path.",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            settings_path=args.settings_path,
            eval_qa_path=args.eval_qa,
            candidate_answer_path=args.candidate_answers,
        )
    )
    print(msg_success("Evaluation Q&A Judge Pipeline Completed"))
