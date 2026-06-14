import json
import tempfile
import unittest
from pathlib import Path

from main_upload_dataset import (
    DEFAULT_EXCLUDE_UPLOAD_KEYS,
    _prepare_split_upload_files,
    _prepare_upload_files,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class UploadDatasetTests(unittest.TestCase):
    def test_default_excludes_item_id_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "all.jsonl"
            work_dir = tmp_path / "work"
            write_jsonl(
                source_path,
                [
                    {
                        "item_id": 1,
                        "question": "q",
                        "source_file": "source.jsonl",
                        "copyright_mitigation": True,
                    }
                ],
            )

            upload_files, counts = _prepare_upload_files(
                [source_path],
                work_dir,
                "qa",
                DEFAULT_EXCLUDE_UPLOAD_KEYS,
            )

            records = read_jsonl(upload_files[0].path)

        self.assertEqual(counts, {"all.jsonl": 1})
        self.assertEqual(upload_files[0].path_in_repo, "all.jsonl")
        self.assertNotIn("item_id", records[0])
        self.assertEqual(records[0]["source_file"], "source.jsonl")
        self.assertTrue(records[0]["copyright_mitigation"])

    def test_custom_exclude_keys_are_removed_from_upload_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "all.jsonl"
            work_dir = tmp_path / "work"
            write_jsonl(
                source_path,
                [
                    {
                        "item_id": 1,
                        "question": "q",
                        "answer": "a",
                        "debug": "internal",
                        "source_file": "source.jsonl",
                    }
                ],
            )

            upload_files, _ = _prepare_upload_files(
                [source_path],
                work_dir,
                "qa",
                ("item_id", "debug", "source_file"),
            )

            records = read_jsonl(upload_files[0].path)

        self.assertEqual(records, [{"question": "q", "answer": "a"}])

    def test_custom_exclude_keys_are_removed_from_split_upload_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_dir = tmp_path / "dataset"
            work_dir = tmp_path / "work"
            write_jsonl(
                dataset_dir / "all.jsonl",
                [
                    {"item_id": 1, "text": "one", "debug": "internal"},
                    {"item_id": 2, "text": "two", "debug": "internal"},
                    {"item_id": 3, "text": "three", "debug": "internal"},
                ],
            )

            upload_files, counts = _prepare_split_upload_files(
                dataset_dir=dataset_dir,
                work_dir=work_dir,
                dataset_kind="qa",
                exclude_upload_keys=("item_id", "debug"),
                validation_ratio=0.34,
                split_seed=1,
            )

            records = []
            for upload_file in upload_files:
                records.extend(read_jsonl(upload_file.path))

        self.assertEqual(counts, {"train.jsonl": 2, "validation.jsonl": 1})
        self.assertEqual([record for record in records if set(record) != {"text"}], [])


if __name__ == "__main__":
    unittest.main()
