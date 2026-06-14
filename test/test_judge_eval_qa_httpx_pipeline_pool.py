import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

import httpx

from pipelines.judge_eval_qa_httpx_pipeline_pool import (
    JudgeEvalQAPipeline,
    entry_cache_key,
    is_entry_processed,
)


class FakeJudgeEvalQAPipeline(JudgeEvalQAPipeline):
    def __init__(self, tmp_path: Path, max_in_flight: int = 2) -> None:
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = prompt_dir / "judge_answer.md"
        prompt_path.write_text(
            "STEP:judge QUESTION:{question} REF:{reference_answer} CAND:{candidate_answer}",
            encoding="utf-8",
        )

        super().__init__(
            {
                "SERVER_URL": "http://fake-server/v1",
                "MODEL_NAME": "fake-judge-model",
                "output_path": str(tmp_path / "output"),
                "prompts": [{"judge_prompt": str(prompt_path)}],
                "batch_size": max_in_flight,
                "max_in_flight": max_in_flight,
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
        item = "bad" if "bad candidate" in prompt else "slow" if "slow candidate" in prompt else "fast"
        self.events.append(
            {
                "event": "start",
                "item": item,
                "time": time.monotonic(),
                "in_flight": self.current_in_flight,
            }
        )
        if item == "slow":
            await asyncio.sleep(0.05)
        else:
            await asyncio.sleep(0.01)
        self.events.append(
            {
                "event": "end",
                "item": item,
                "time": time.monotonic(),
                "in_flight": self.current_in_flight,
            }
        )

        if item == "bad":
            content = ""
        elif item == "slow":
            content = (
                "<judge_score>3</judge_score>"
                "<judge_label>partially_correct</judge_label>"
                "<judge_reason>一部は合っているが不足があります。</judge_reason>"
            )
        else:
            content = (
                "<judge_score>5</judge_score>"
                "<judge_label>correct</judge_label>"
                "<judge_reason>基準回答と同じ内容です。</judge_reason>"
            )

        return httpx.Response(
            200,
            request=httpx.Request("POST", "http://fake-server/v1/chat/completions"),
            content=json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8"),
        )


class JudgeEvalQAPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_judge_output_is_parsed_and_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = FakeJudgeEvalQAPipeline(Path(tmpdir), max_in_flight=1)
            all_path = Path(tmpdir) / "all.jsonl"
            failure_path = Path(tmpdir) / "all.failures.jsonl"
            cache_path = Path(tmpdir) / "cache_processed_ids.txt"
            try:
                stats = await pipeline.judge_qa_async(
                    [
                        {
                            "id": "qa-1",
                            "chunk_index": 0,
                            "question": "question",
                            "reference_answer": "answer",
                            "candidate_answer": "fast candidate",
                            "cache_key": "qa-1\t0",
                        }
                    ],
                    all_path,
                    failure_path,
                    cache_path,
                )
                saved = json.loads(all_path.read_text(encoding="utf-8").strip())
                cached = cache_path.read_text(encoding="utf-8").strip()
            finally:
                await pipeline.aclose()

        self.assertEqual(stats["saved_records"], 1)
        self.assertEqual(saved["judge_score"], 5)
        self.assertEqual(saved["judge_label"], "correct")
        self.assertEqual(saved["judge_reason"], "基準回答と同じ内容です。")
        self.assertEqual(saved["judge_model"], "fake-judge-model")
        self.assertEqual(cached, "qa-1\t0")

    async def test_one_failed_item_does_not_stop_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = FakeJudgeEvalQAPipeline(Path(tmpdir), max_in_flight=2)
            all_path = Path(tmpdir) / "all.jsonl"
            failure_path = Path(tmpdir) / "all.failures.jsonl"
            cache_path = Path(tmpdir) / "cache_processed_ids.txt"
            try:
                stats = await pipeline.judge_qa_async(
                    [
                        {
                            "id": "qa-bad",
                            "chunk_index": 0,
                            "question": "question",
                            "reference_answer": "answer",
                            "candidate_answer": "bad candidate",
                            "cache_key": "qa-bad\t0",
                        },
                        {
                            "id": "qa-fast",
                            "chunk_index": 0,
                            "question": "question",
                            "reference_answer": "answer",
                            "candidate_answer": "fast candidate",
                            "cache_key": "qa-fast\t0",
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
        self.assertEqual(saved["id"], "qa-fast")
        self.assertEqual(failure["id"], "qa-bad")
        self.assertEqual(failure["failed_step"], 1)
        self.assertLessEqual(pipeline.max_observed_in_flight, 2)

    async def test_judge_dataset_writes_stats_and_skips_processed_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            eval_path = tmp_path / "eval.jsonl"
            answers_path = tmp_path / "answers.jsonl"
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            (output_dir / "cache_processed_ids.txt").write_text("qa-1\t0\n", encoding="utf-8")
            eval_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "qa-1",
                                "chunk_index": 0,
                                "question": "processed",
                                "answer": "answer",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "id": "qa-2",
                                "chunk_index": 1,
                                "question": "pending",
                                "answer": "answer",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            answers_path.write_text(
                json.dumps(
                    {
                        "id": "qa-2",
                        "chunk_index": 1,
                        "answer": "slow candidate",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            pipeline = FakeJudgeEvalQAPipeline(tmp_path, max_in_flight=1)
            try:
                stats = await pipeline.judge_dataset(eval_path, answers_path)
                saved = json.loads((output_dir / "all.jsonl").read_text(encoding="utf-8").strip())
                stats_file = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
            finally:
                await pipeline.aclose()

        self.assertEqual(stats["saved_records"], 1)
        self.assertEqual(stats["pending_records"], 1)
        self.assertEqual(saved["id"], "qa-2")
        self.assertEqual(saved["chunk_index"], 1)
        self.assertEqual(saved["judge_label"], "partially_correct")
        self.assertEqual(stats_file["saved_records"], 1)


class JudgeEvalQAInputTests(unittest.TestCase):
    def test_entry_cache_key_uses_id_and_chunk_index(self) -> None:
        self.assertEqual(entry_cache_key("qa-1", 3), "qa-1\t3")
        self.assertEqual(entry_cache_key("qa-1", None), "qa-1")

    def test_processed_cache_keeps_unfinished_chunk_pending(self) -> None:
        processed_keys = {"qa-1\t0"}

        self.assertTrue(is_entry_processed("qa-1", 0, processed_keys))
        self.assertFalse(is_entry_processed("qa-1", 1, processed_keys))

    def test_prepare_candidates_matches_by_id_and_chunk_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = FakeJudgeEvalQAPipeline(Path(tmpdir))
            try:
                eval_entries = [
                    {"id": "qa-1", "chunk_index": 0, "question": "q0", "answer": "ref0"},
                    {"id": "qa-1", "chunk_index": 1, "question": "q1", "answer": "ref1"},
                ]
                answer_index = pipeline.build_answer_index(
                    [
                        {"id": "qa-1", "chunk_index": 1, "answer": "candidate1"},
                    ]
                )
                candidates = pipeline.prepare_candidates(eval_entries, answer_index, set())
            finally:
                asyncio.run(pipeline.aclose())

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["chunk_index"], 1)
        self.assertEqual(candidates[0]["candidate_answer"], "candidate1")
        self.assertEqual(candidates[0]["cache_key"], "qa-1\t1")


if __name__ == "__main__":
    unittest.main()
