import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from commons.util_settings import load_settings
from commons.utils_msg import msg_error, msg_info, msg_success


DEFAULT_SETTINGS_PATH = "./yamls/cpt_wiki_settings_format.yaml"
DEFAULT_DATASET_PATH = "./test_output/cpt/wiki"


@dataclass(frozen=True)
class UploadFile:
    path: Path
    path_in_repo: str


def _resolve_dataset_dir(dataset_path: str | None, settings_path: str | None) -> Path:
    if dataset_path:
        return Path(dataset_path).expanduser().resolve()

    if settings_path:
        settings = load_settings(Path(settings_path))
        return Path(settings.get("output_path", DEFAULT_DATASET_PATH)).expanduser().resolve()

    return Path(DEFAULT_DATASET_PATH).expanduser().resolve()


def _iter_jsonl(file_path: Path) -> Iterable[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row in {file_path}: line={line_no} error={exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row must be an object in {file_path}: line={line_no}")
            yield obj


def _count_jsonl_records(file_path: Path) -> int:
    return sum(1 for _ in _iter_jsonl(file_path))


def _is_copyright_mitigated(record: dict) -> bool:
    value = record.get("copyright_mitigation")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def _write_upload_jsonl(source_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for record in _iter_jsonl(source_path):
            if not _is_copyright_mitigated(record):
                continue
            upload_record = dict(record)
            upload_record.pop("source_file", None)
            upload_record.pop("copyright_mitigation", None)
            json.dump(upload_record, f, ensure_ascii=False)
            f.write("\n")
            count += 1
    return count


def _collect_upload_files(dataset_dir: Path, include_splits: bool) -> list[Path]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    required_files = [dataset_dir / "all.jsonl"]
    optional_files = []
    if include_splits:
        optional_files.extend([dataset_dir / "train.jsonl", dataset_dir / "validation.jsonl"])

    missing = [path for path in required_files if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required dataset file is missing: {missing_text}")

    files = [path for path in required_files + optional_files if path.is_file()]

    return files


def _prepare_upload_files(files: list[Path], work_dir: Path) -> tuple[list[UploadFile], dict[str, int]]:
    upload_files: list[UploadFile] = []
    counts: dict[str, int] = {}
    for file_path in files:
        if file_path.suffix == ".jsonl":
            upload_path = work_dir / file_path.name
            counts[file_path.name] = _write_upload_jsonl(file_path, upload_path)
            upload_files.append(UploadFile(path=upload_path, path_in_repo=file_path.name))
        elif file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                json.load(f)
            upload_files.append(UploadFile(path=file_path, path_in_repo=file_path.name))
    return upload_files, counts


def _upload_files(
    repo_id: str,
    token: str,
    files: list[UploadFile],
    private: bool,
    commit_message: str,
) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required. Install it with `pip install huggingface-hub` "
            "or add it to the project dependencies."
        ) from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    for upload_file in files:
        print(msg_info(f"Uploading {upload_file.path} -> {repo_id}/{upload_file.path_in_repo}"))
        api.upload_file(
            path_or_fileobj=str(upload_file.path),
            path_in_repo=upload_file.path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )

    return f"https://huggingface.co/datasets/{repo_id}"


def main(
    repo_id: str | None,
    hf_token: str | None,
    dataset_path: str | None,
    settings_path: str | None,
    private: bool,
    include_splits: bool,
    dry_run: bool,
    commit_message: str,
) -> None:
    if not repo_id:
        print(msg_error("repo_id is required."), file=sys.stderr)
        sys.exit(1)
    if not hf_token and not dry_run:
        print(msg_error("hf_token is required unless --dry-run is set."), file=sys.stderr)
        sys.exit(1)

    dataset_dir = _resolve_dataset_dir(dataset_path, settings_path)
    files = _collect_upload_files(dataset_dir, include_splits=include_splits)

    with tempfile.TemporaryDirectory(prefix="cpt_upload_") as temp_dir:
        upload_files, counts = _prepare_upload_files(files, Path(temp_dir))

        print(msg_info(f"dataset_dir={dataset_dir}"))
        for upload_file in upload_files:
            detail = f" records={counts[upload_file.path_in_repo]}" if upload_file.path_in_repo in counts else ""
            print(msg_info(f"upload_file={upload_file.path_in_repo}{detail}"))

        if dry_run:
            print(msg_success("Dry run completed. No files were uploaded."))
            return

        url = _upload_files(
            repo_id=repo_id,
            token=hf_token or "",
            files=upload_files,
            private=private,
            commit_message=commit_message,
        )
    print(msg_success(f"Uploaded CPT dataset: {url}"))


if __name__ == "__main__":
    print(msg_success("CPT Dataset Upload Pipeline Started"))

    parser = argparse.ArgumentParser(
        description="Upload a CPT dataset created by main_create_cpt_dataset.py to Hugging Face Datasets."
    )
    parser.add_argument(
        "-r",
        "--repo_id",
        required=True,
        help="Hugging Face dataset repository ID. Example: username/dataset_name",
    )
    parser.add_argument(
        "-t",
        "--hf_token",
        default=None,
        help="Hugging Face access token with write permission.",
    )
    parser.add_argument(
        "-d",
        "--dataset_path",
        default=None,
        help="Path to the CPT dataset output directory. Defaults to YAML output_path.",
    )
    parser.add_argument(
        "-p",
        "--settings_path",
        default=DEFAULT_SETTINGS_PATH,
        help="Path to the CPT settings YAML file used to read output_path.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hugging Face dataset repository as private.",
    )
    parser.add_argument(
        "--include-splits",
        action="store_true",
        help="Also upload train.jsonl and validation.jsonl.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and print upload targets without uploading.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload CPT dataset",
        help="Commit message used for uploaded files.",
    )

    args = parser.parse_args()
    main(
        repo_id=args.repo_id,
        hf_token=args.hf_token,
        dataset_path=args.dataset_path,
        settings_path=args.settings_path,
        private=args.private,
        include_splits=args.include_splits,
        dry_run=args.dry_run,
        commit_message=args.commit_message,
    )

    print(msg_success("CPT Dataset Upload Pipeline Completed"))




"""
uv run python main_upload_cpt_dataset.py \
  -r ikedachin/your_cpt_dataset_name \
  -t hf_xxxxxxxxxxxxxxxxx \
  --include-splits # Optional: --dataset-path /path/to/cpt/dataset \
"""
