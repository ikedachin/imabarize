import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

import httpx

from pipelines.create_rl_qa_httpx_pipeline_pool import (
    RLQAPipeline,
    entry_cache_key,
    is_entry_processed,
)


class FakeRLQAPipeline(RLQAPipeline):
    def __init__(self, tmp_path: Path, max_in_flight: int = 2, sample_size: int = 500) -> None:
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompts = {
            "question_prompt": "STEP:question REF:{text} title={title} token={random_token}",
            "blind_answer_prompt": "STEP:blind QUESTION:{question}",
            "grounded_answer_prompt": "STEP:grounded REF:{text} QUESTION:{question}",
            "choices_prompt": (
                "STEP:choices QUESTION:{question} BLIND:{blind_answer} "
                "GROUNDED:{grounded_answer} EVIDENCE:{evidence}"
            ),
        }
        prompt_settings: List[Dict[str, str]] = []
        for name, body in prompts.items():
            path = prompt_dir / f"{name}.txt"
            path.write_text(body, encoding="utf-8")
            prompt_settings.append({name: str(path)})

        super().__init__(
            {
                "SERVER_URL": "http://fake-server/v1",
                "MODEL_NAME": "fake-model",
                "output_path": str(tmp_path / "output"),
                "prompts": prompt_settings,
                "batch_size": max_in_flight,
                "max_in_flight": max_in_flight,
                "sample_size": sample_size,
                "max_retries": 1,
                "wait_seconds": 0,
                "retry_jitter_seconds": 0,
            }
        )
        self.payloads: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []

    async def _post_chat_completion(self, payload: Dict[str, Any]) -> httpx.Response:
        self.payloads.append(payload)
        prompt = payload["messages"][0]["content"]
        step = prompt.split("STEP:", 1)[1].split(" ", 1)[0]
        item = "bad" if "bad source" in prompt or "bad question" in prompt else "slow" if "slow source" in prompt else "fast"
        self.events.append(
            {
                "event": "start",
                "item": item,
                "step": step,
                "time": time.monotonic(),
                "in_flight": self.current_in_flight,
            }
        )
        if item == "slow" and step == "question":
            await asyncio.sleep(0.05)
        else:
            await asyncio.sleep(0.01)
        self.events.append(
            {
                "event": "end",
                "item": item,
                "step": step,
                "time": time.monotonic(),
                "in_flight": self.current_in_flight,
            }
        )

        if item == "bad" and step == "blind":
            content = ""
        elif step == "question":
            content = f"<question>{item} question</question>"
        elif step == "blind":
            content = "<blind_answer>wrong-answer</blind_answer><blind_confidence>中</blind_confidence>"
        elif step == "grounded":
            content = "<grounded_answer>grounded-answer</grounded_answer><evidence>source evidence</evidence>"
        elif step == "choices":
            content = (
                "<choice_a>grounded-answer</choice_a>"
                "<choice_b>wrong-answer</choice_b>"
                "<choice_c>near miss 1</choice_c>"
                "<choice_d>near miss 2</choice_d>"
                "<correct_label>A</correct_label>"
                "<correct_answer>grounded-answer</correct_answer>"
                "<difficulty>borderline</difficulty>"
                "<rl_suitability>accepted</rl_suitability>"
                "<rejection_reason></rejection_reason>"
            )
        else:
            content = ""

        return httpx.Response(
            200,
            request=httpx.Request("POST", "http://fake-server/v1/chat/completions"),
            content=json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8"),
        )


class RLQAPipelinePoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_four_steps_pass_expected_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = FakeRLQAPipeline(Path(tmpdir), max_in_flight=1)
            all_path = Path(tmpdir) / "all.jsonl"
            failure_path = Path(tmpdir) / "all.failures.jsonl"
            cache_path = Path(tmpdir) / "cache_processed_ids.txt"
            try:
                await pipeline.create_rl_qa_async(
                    [
                        {
                            "id": "article-1",
                            "chunk_index": 0,
                            "title": "title",
                            "reference_text": "fast source",
                            "cache_key": "article-1\t0",
                        }
                    ],
                    all_path,
                    failure_path,
                    cache_path,
                )
                prompts = [payload["messages"][0]["content"] for payload in pipeline.payloads]
                saved = json.loads(all_path.read_text(encoding="utf-8").strip())
            finally:
                await pipeline.aclose()

        self.assertIn("REF:fast source", prompts[0])
        self.assertNotIn("REF:", prompts[1])
        self.assertIn("REF:fast source", prompts[2])
        self.assertIn("BLIND:wrong-answer", prompts[3])

        self.assertEqual(len(saved["choices"]), 4)
        self.assertEqual(saved["correct_label"], "A")
        self.assertEqual(saved["correct_answer"], "grounded-answer")
        self.assertEqual(saved["choices"][1]["text"], "wrong-answer")
        self.assertEqual(saved["rl_suitability"], "accepted")
        self.assertLessEqual(pipeline.max_observed_in_flight, 1)

    async def test_one_failed_item_does_not_stop_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = FakeRLQAPipeline(Path(tmpdir), max_in_flight=2)
            all_path = Path(tmpdir) / "all.jsonl"
            failure_path = Path(tmpdir) / "all.failures.jsonl"
            cache_path = Path(tmpdir) / "cache_processed_ids.txt"
            try:
                stats = await pipeline.create_rl_qa_async(
                    [
                        {
                            "id": "article-bad",
                            "chunk_index": 0,
                            "title": "bad",
                            "reference_text": "bad source",
                            "cache_key": "article-bad\t0",
                        },
                        {
                            "id": "article-fast",
                            "chunk_index": 0,
                            "title": "fast",
                            "reference_text": "fast source",
                            "cache_key": "article-fast\t0",
                        },
                    ],
                    all_path,
                    failure_path,
                    cache_path,
                )
                saved = json.loads(all_path.read_text(encoding="utf-8").strip())
                failure = json.loads(failure_path.read_text(encoding="utf-8").strip())
            finally:
                await pipeline.aclose()

        self.assertEqual(stats["saved_records"], 1)
        self.assertEqual(stats["failed_items"], 1)
        self.assertEqual(saved["id"], "article-fast")
        self.assertEqual(failure["id"], "article-bad")
        self.assertEqual(failure["failed_step"], 2)
        self.assertLessEqual(pipeline.max_observed_in_flight, 2)


class RLQAPipelineInputTests(unittest.TestCase):
    def test_random_sample_is_seeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = FakeRLQAPipeline(Path(tmpdir), sample_size=3)
            try:
                candidates = [{"id": str(i)} for i in range(10)]
                first = pipeline.sample_candidates(candidates)
                second = pipeline.sample_candidates(candidates)
            finally:
                asyncio.run(pipeline.aclose())

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertNotEqual(first, candidates[:3])

    def test_entry_cache_key_uses_id_and_chunk_index(self) -> None:
        self.assertEqual(entry_cache_key("article-1", 3), "article-1\t3")
        self.assertEqual(entry_cache_key("article-1", None), "article-1")

    def test_processed_cache_keeps_unfinished_chunk_pending(self) -> None:
        processed_keys = {"article-1\t0"}

        self.assertTrue(is_entry_processed("article-1", 0, processed_keys))
        self.assertFalse(is_entry_processed("article-1", 1, processed_keys))

    def test_prepare_candidates_skips_processed_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = FakeRLQAPipeline(Path(tmpdir))
            try:
                entries = [
                    {"id": "article-1", "chunk_index": 0, "text": "processed", "source_file": "source.jsonl"},
                    {"id": "article-1", "chunk_index": 1, "text": "pending", "source_file": "source.jsonl"},
                    {"id": "article-2", "chunk_index": 0, "text": "", "source_file": "source.jsonl"},
                ]
                candidates = pipeline.prepare_candidates(entries, {"article-1\t0"})
            finally:
                asyncio.run(pipeline.aclose())

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["id"], "article-1")
        self.assertEqual(candidates[0]["chunk_index"], 1)
        self.assertEqual(candidates[0]["cache_key"], "article-1\t1")


if __name__ == "__main__":
    unittest.main()
