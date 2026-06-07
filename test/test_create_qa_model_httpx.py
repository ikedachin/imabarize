import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

import httpx

from pipelines.create_qa_model_httpx import QAPipeline


class FakeRollingPipeline(QAPipeline):
    def __init__(self, tmp_path: Path, max_in_flight: int = 2) -> None:
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompts = {
            "question_prompt": "STEP:question item={text} token={random_token}",
            "answer_prompt": "STEP:answer item={text} question={question}",
            "thinking_prompt": "STEP:thinking item={text} question={question} answer={answer}",
            "refine_answer_prompt": "STEP:refine item={text} question={question} answer={answer}",
            "eval_prompt": "STEP:eval think={think} answer={answer}",
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
                "max_retries": 1,
                "wait_seconds": 0,
                "retry_jitter_seconds": 0,
            }
        )
        self.events: List[Dict[str, Any]] = []

    async def _post_chat_completion(self, payload: Dict[str, Any]) -> httpx.Response:
        prompt = payload["messages"][0]["content"]
        item = "slow" if "item=slow" in prompt else "bad" if "item=bad" in prompt else "fast"
        step = prompt.split("STEP:", 1)[1].split(" ", 1)[0]
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
            await asyncio.sleep(0.2)
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

        if item == "bad" and step == "answer":
            content = ""
        elif step == "question":
            content = f"<question>{item} question</question>"
        elif step == "answer":
            content = f"<answer>{item} answer</answer>"
        elif step == "thinking":
            content = f"<think>{item} thinking</think>"
        elif step == "refine":
            content = f"<think>{item} refined thinking</think>\n\n{item} refined answer"
        elif step == "eval":
            content = "<eval>5</eval>"
        else:
            content = ""

        return httpx.Response(
            200,
            request=httpx.Request("POST", "http://fake-server/v1/chat/completions"),
            content=json.dumps(
                {"choices": [{"message": {"content": content}}]},
            ).encode("utf-8"),
        )


class RollingPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_item_reaches_step5_before_slow_step1_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = FakeRollingPipeline(Path(tmpdir), max_in_flight=2)
            completed_order: List[int] = []

            async def on_result(item_id: int, result: Dict[str, Any]) -> None:
                completed_order.append(item_id)

            try:
                results = await pipeline.create_qa_batch_async(
                    ["slow", "fast"],
                    batch_size=2,
                    on_result=on_result,
                )
            finally:
                await pipeline.aclose()

        slow_question_end = next(
            event["time"]
            for event in pipeline.events
            if event["event"] == "end" and event["item"] == "slow" and event["step"] == "question"
        )
        fast_eval_end = next(
            event["time"]
            for event in pipeline.events
            if event["event"] == "end" and event["item"] == "fast" and event["step"] == "eval"
        )

        self.assertLess(fast_eval_end, slow_question_end)
        self.assertEqual(completed_order[0], 1)
        self.assertEqual([result["item_id"] for result in results], [0, 1])
        self.assertEqual(results[1]["answer"], "fast refined answer")
        self.assertLessEqual(pipeline.max_observed_in_flight, 2)

    async def test_one_failed_item_does_not_stop_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = FakeRollingPipeline(Path(tmpdir), max_in_flight=2)
            try:
                results = await pipeline.create_qa_batch_async(["bad", "fast"], batch_size=2)
            finally:
                await pipeline.aclose()

        self.assertTrue(results[0]["failed"])
        self.assertEqual(results[0]["failed_step"], 2)
        self.assertFalse(results[1].get("failed", False))
        self.assertEqual(results[1]["item_id"], 1)
        self.assertEqual(results[1]["eval"], "5")
        self.assertLessEqual(pipeline.max_observed_in_flight, 2)


if __name__ == "__main__":
    unittest.main()
