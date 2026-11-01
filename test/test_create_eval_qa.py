import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pipelines.create_qa_model import PipelineJob
from main_create_eval_qa import EvalQAPipeline, main

from main_create_eval_qa import parse_sample_size, select_sampled_items


class EvalQASamplingTests(unittest.TestCase):
    def test_parse_sample_size_allows_none_or_non_positive_as_all(self) -> None:
        self.assertIsNone(parse_sample_size(None))
        self.assertIsNone(parse_sample_size(0))
        self.assertIsNone(parse_sample_size(-1))
        self.assertEqual(parse_sample_size("3"), 3)

    def test_select_sampled_items_is_seeded_and_limited(self) -> None:
        items = [{"id": str(i)} for i in range(10)]

        first = select_sampled_items(items, sample_size=4, seed=42)
        second = select_sampled_items(items, sample_size=4, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertNotEqual(first, items[:4])

    def test_select_sampled_items_returns_all_when_limit_is_none_or_large(self) -> None:
        items = [{"id": str(i)} for i in range(3)]

        self.assertEqual(select_sampled_items(items, sample_size=None, seed=42), items)
        self.assertEqual(select_sampled_items(items, sample_size=10, seed=42), items)


class EvalQAPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_qa_only_and_reject_unusable_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            question = Path(tmp) / "q.md"
            answer = Path(tmp) / "a.md"
            question.write_text("{text}", encoding="utf-8")
            answer.write_text("{question}", encoding="utf-8")
            pipeline = EvalQAPipeline({"output_path": tmp, "MODEL_NAME": "model", "prompts": [{"question_prompt": str(question)}, {"answer_prompt": str(answer)}, {"thinking_prompt": "/missing"}]})
            try:
                pipeline._infer_text_async = AsyncMock(side_effect=["<question>問い</question>", "<answer>答えなんよ。</answer>"])
                results = await pipeline.create_qa_batch_async(["本文"], batch_size=1)
                self.assertEqual(pipeline._infer_text_async.await_count, 2)
                self.assertNotIn("thinking", results[0])
                self.assertEqual(results[0]["messages"][1]["content"], "答えなんよ。")
                pipeline._infer_text_async = AsyncMock(return_value="<question>作成不可</question>")
                with self.assertRaises(ValueError):
                    await pipeline._run_one_step(PipelineJob(0, 1, {"text": "本文"}))
            finally:
                await pipeline.aclose()

    async def test_mixed_sources_sample_before_resume_and_retry_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "output"
            source.mkdir()
            for i in range(3):
                (source / f"{i}.txt").write_text(f"text-{i}", encoding="utf-8")
            (source / "data.jsonl").write_text("".join(json.dumps({"id": str(i), "body": f"json-{i}"}) + "\n" for i in range(7)), encoding="utf-8")
            settings = {"output_path": str(output), "sample_size": 5, "seed": 42}
            calls = []
            class FakePipeline:
                def __init__(self, settings):
                    self.output_dir = Path(settings["output_path"])
                    self.output_dir.mkdir(exist_ok=True)
                    self.max_in_flight = 1
                async def aclose(self):
                    pass
                async def create_qa_batch_async(self, texts, batch_size, on_result):
                    calls.append(list(texts))
                    for i, text in enumerate(texts):
                        failed = len(calls) == 1 and i == 0
                        await on_result(i, {"failed": True} if failed else {"question": "Q", "answer": "A", "messages": [], "qa_generator": "model"})
                    return []
                def append_jsonl(self, path, result):
                    with path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(result) + "\n")
                def append_failure_jsonl(self, path, result):
                    with path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(result) + "\n")
            with patch("main_create_eval_qa.load_settings", return_value=settings), patch("main_create_eval_qa.EvalQAPipeline", FakePipeline):
                await main("settings", str(source), "body", 0)
                await main("settings", str(source), "body", 0)
                await main("settings", str(source), "body", 0)
            self.assertEqual(len(calls[0]), 5)
            self.assertTrue(any(t.startswith("text-") for t in calls[0]))
            self.assertTrue(any(t.startswith("json-") for t in calls[0]))
            self.assertEqual(calls[1], calls[0][:1])
            self.assertEqual(calls[2], [])
            self.assertEqual(len((output / "cache_eval_qa.txt").read_text().splitlines()), 5)


if __name__ == "__main__":
    unittest.main()
