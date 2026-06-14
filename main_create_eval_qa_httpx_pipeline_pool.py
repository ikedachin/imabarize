import argparse
import asyncio

from commons.utils_msg import msg_success
from main_create_imabari_qa_httpx_pipeline_pool import main


if __name__ == "__main__":
    print(msg_success("Evaluation Q&A Creation Pipeline Started"))

    parser = argparse.ArgumentParser(
        description="Create evaluation Q&A from text, markdown, and json files (httpx async pipeline pool)."
    )
    parser.add_argument(
        "-p",
        "--settings_path",
        nargs="?",
        default="./yamls/eval_qa_settings_format.yaml",
        help="Path to the settings YAML file",
    )
    parser.add_argument(
        "-s",
        "--source",
        nargs="?",
        default=None,
        help="Path to a file or a directory containing sources",
    )
    parser.add_argument(
        "-t",
        "--target_key",
        type=str,
        default=None,
        help="Target key to extract from JSON/JSONL files",
    )
    parser.add_argument(
        "-i",
        "--start_index",
        type=int,
        default=0,
        help="Start index for resuming processing",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            settings_path=args.settings_path,
            source_path=args.source,
            target_key=args.target_key,
            start_index=args.start_index,
        )
    )

    print(msg_success("Evaluation Q&A Creation Pipeline Completed"))
