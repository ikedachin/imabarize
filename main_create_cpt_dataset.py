import argparse
import sys
from pathlib import Path

from commons.util_settings import load_settings
from commons.utils_msg import msg_error, msg_info, msg_success
from pipelines.create_cpt_dataset import CPTDatasetPipeline


def main(settings_path: str | None, source_path: str | None) -> None:
    if settings_path is None:
        print(msg_error("settings_path is required."), file=sys.stderr)
        sys.exit(1)
    if source_path is None:
        print(msg_error("source path is required."), file=sys.stderr)
        sys.exit(1)

    settings = load_settings(Path(settings_path))
    source = Path(source_path).expanduser().resolve()

    pipeline = CPTDatasetPipeline(settings)
    train_path, validation_path, stats = pipeline.build_dataset(source)

    print(msg_info(f"entries={stats['entries']} chunks={stats['chunks']}"))
    print(msg_info(f"train={train_path}"))
    if validation_path:
        print(msg_info(f"validation={validation_path}"))


if __name__ == "__main__":
    print(msg_success("CPT Dataset Creation Pipeline Started"))

    parser = argparse.ArgumentParser(description="Create a CPT dataset from text, markdown, and json files.")
    parser.add_argument(
        "-p",
        "--settings_path",
        nargs="?",
        default="./yamls/cpt_wiki_settings_format.yaml",
        help="Path to the settings YAML file",
    )
    parser.add_argument(
        "-s",
        "--source",
        nargs="?",
        default="./test_source/wiki/raw.jsonl",
        help="Path to a file or a directory containing sources",
    )

    args = parser.parse_args()
    main(settings_path=args.settings_path, source_path=args.source)

    print(msg_success("CPT Dataset Creation Pipeline Completed"))
