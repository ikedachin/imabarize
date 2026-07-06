import argparse
import asyncio
import sys
from pathlib import Path

from commons.util_settings import load_settings
from commons.utils_msg import msg_error, msg_info, msg_success
from pipelines.create_rl_qa import RLQAPipeline


async def main(settings_path: str | None, source_path: str | None) -> None:
    if settings_path is None:
        print(msg_error("settings_path is required."), file=sys.stderr)
        sys.exit(1)

    settings = load_settings(Path(settings_path))
    configured_source = source_path or settings.get("source_path") or "./test_output/cpt/wiki/all.jsonl"
    source = Path(configured_source).expanduser().resolve()

    pipeline = RLQAPipeline(settings)
    print(
        msg_info(
            f"Async request concurrency max_in_flight={pipeline.max_in_flight}, "
            f"input_window_hint={pipeline.pipeline_batch_size}"
        )
    )
    try:
        stats = await pipeline.build_dataset(source)
        print(msg_success(f"GRPO RL QA dataset saved to: {pipeline.output_dir}"))
        print(msg_info(f"Stats: {stats}"))
    finally:
        await pipeline.aclose()


if __name__ == "__main__":
    print(msg_success("GRPO RL Q&A Creation Pipeline Started"))

    parser = argparse.ArgumentParser(
        description="Create four-choice RL/GRPO Q&A dataset from JSON/JSONL sources (httpx async pipeline pool)."
    )
    parser.add_argument(
        "-p",
        "--settings_path",
        nargs="?",
        default="./yamls/create_grpo_qa_settings_format.yaml",
        help="Path to the settings YAML file",
    )
    parser.add_argument(
        "-s",
        "--source",
        nargs="?",
        default=None,
        help="Path to source JSON/JSONL. Defaults to settings source_path or ./test_output/cpt/wiki/all.jsonl.",
    )
    args = parser.parse_args()

    asyncio.run(main(settings_path=args.settings_path, source_path=args.source))
    print(msg_success("GRPO RL Q&A Creation Pipeline Completed"))
