import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
from datasets import Dataset

from tool_upload_for_llmjp4 import DEFAULT_REPO, prepare, read_records, upload


def sample(index):
    return {
        "qa_id": f"converted-{index}", "source_qa_id": "original", "id": "article", "chunk_index": 0,
        "question": "質問？", "answer": "回答です。", "thinking": "説明です。",
        "reasoning_effort": "low", "eval": None,
        "messages": [{"role": "user", "content": "質問？"},
                     {"role": "assistant", "content": "回答です。", "thinking": "説明です。"}],
        "chat_template_kwargs": {"reasoning_effort": "low", "conversation_start_date": "2026-09-11"},
        "thinking_tokens": 5, "thinking_generator": "generator",
        "source_metadata": {"qa_id": "original", "id": "article", "chunk_index": 0},
    }


class StandardJapaneseUploadTests(unittest.TestCase):
    def test_evaluated_generated_records_preserve_score_and_source_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [sample(i) for i in range(2)]
            for row in rows:
                row["eval"] = "5"
                row["source_metadata"]["qa_id"] = None
            source = root / "input.jsonl"
            source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            prepare(source, root / "output", allow_evaluated=True)
            for split in ("train", "validation"):
                restored = Dataset(pq.read_table(root / "output/data" / f"{split}-00000-of-00001.parquet"))
                self.assertEqual(restored[0]["eval"], "5")
                self.assertEqual(restored[0]["source_metadata"]["qa_id"], "original")

    def test_roundtrip_without_readme_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.jsonl"
            original = "\n".join(json.dumps(sample(i), ensure_ascii=False) for i in range(10))
            source.write_text(original, encoding="utf-8")
            with patch("tool_upload_standard_japanese_llmjp4.HfApi", side_effect=AssertionError("network")):
                counts = prepare(source, root / "output")
            self.assertEqual(counts, {"train": 9, "validation": 1})
            self.assertEqual(source.read_text(encoding="utf-8"), original)
            actual = []
            for split in counts:
                dataset = Dataset(pq.read_table(root / "output/data" / f"{split}-00000-of-00001.parquet"))
                actual.extend(dataset)
                self.assertEqual(dataset.features["eval"].dtype, "string")
                self.assertIsInstance(dataset[0]["messages"][0], dict)
            self.assertEqual({row["qa_id"] for row in actual}, {f"converted-{i}" for i in range(10)})
            self.assertTrue(all(row["eval"] is None and "source_qa_id" not in row for row in actual))
            self.assertFalse((root / "output/README.md").exists())

    def test_rejects_information_loss_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.jsonl"
            for mutation in ({"id": "wrong"}, {"qa_id": "converted-0"}, {"unknown": 1},
                             {"eval": "5"}, {"messages": []}):
                with self.subTest(mutation=mutation):
                    changed = sample(1) | mutation
                    path.write_text(json.dumps(sample(0)) + "\n" + json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_records(path)

    def test_upload_uses_only_prepared_files_in_one_commit(self):
        with patch("tool_upload_standard_japanese_llmjp4.HfApi") as api_class, \
                patch("tool_upload_standard_japanese_llmjp4.load_dotenv"):
            api = api_class.return_value
            api.upload_folder.return_value.commit_url = "https://example.com/commit"
            result = upload(Path("prepared"), DEFAULT_REPO, private=True)
            api.create_repo.assert_called_once_with(repo_id=DEFAULT_REPO, repo_type="dataset", private=True, exist_ok=True)
            args = api.upload_folder.call_args.kwargs
            self.assertEqual(args["repo_id"], DEFAULT_REPO)
            self.assertEqual(args["allow_patterns"], [
                "data/train-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet",
            ])
            self.assertNotIn("delete_patterns", args)
            self.assertEqual(result, "https://example.com/commit")

    def test_upload_loads_dotenv_and_preserves_environment_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("HF_TOKEN=test-dotenv-token\n", encoding="utf-8")
            for environment, expected in (({}, "test-dotenv-token"),
                                          ({"HF_TOKEN": "test-env-token"}, "test-env-token")):
                with self.subTest(environment=environment), \
                        patch.dict(os.environ, environment, clear=True), \
                        patch("tool_upload_standard_japanese_llmjp4.ROOT", root), \
                        patch("tool_upload_standard_japanese_llmjp4.HfApi") as api_class:
                    upload(root, DEFAULT_REPO)
                    api_class.assert_called_once_with(token=expected)


if __name__ == "__main__":
    unittest.main()
