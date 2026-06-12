import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

import httpx

from pipelines.create_cpt_dataset_httpx_pipeline_pool import CPTDatasetPipelinePool


class FakeCPTPipelinePool(CPTDatasetPipelinePool):
    def __init__(
        self,
        tmp_path: Path,
        max_in_flight: int = 2,
        copyright_mitigation: bool = True,
        cpt_enable_thinking: Any = None,
        copyright_mitigation_failure_policy: str = "fail",
    ) -> None:
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompts = {
            "to_bullet_points_prompt": "BULLET:{text}",
            "bullet_points_to_text_prompt": "RECON:{text}",
        }
        prompt_settings: List[Dict[str, str]] = []
        for name, body in prompts.items():
            path = prompt_dir / f"{name}.txt"
            path.write_text(body, encoding="utf-8")
            prompt_settings.append({name: str(path)})

        super().__init__(
            {
                "SERVER_URL": "http://fake-server/v1",
                "MODEL_NAME": "fake-cpt-model",
                "output_path": str(tmp_path / "output"),
                "target_key": "content",
                "title_key": "title",
                "id_key": "id",
                "text_key": "text",
                "include_title": False,
                "min_chars": 1,
                "max_chars": 1000,
                "overlap_chars": 0,
                "train_ratio": 1.0,
                "shuffle": False,
                "deduplicate": False,
                "copyright_mitigation": copyright_mitigation,
                "copyright_mitigation_failure_policy": copyright_mitigation_failure_policy,
                "cpt_enable_thinking": cpt_enable_thinking,
                "prompts": prompt_settings,
                "batch_size": max_in_flight,
                "max_in_flight": max_in_flight,
                "max_retries": 1,
                "wait_seconds": 0,
                "retry_jitter_seconds": 0,
            }
        )
        self.events: List[Dict[str, Any]] = []
        self.payloads: List[Dict[str, Any]] = []

    async def _post_chat_completion(self, payload: Dict[str, Any]) -> httpx.Response:
        self.payloads.append(payload)
        prompt = payload["messages"][0]["content"]
        step = "bullet" if prompt.startswith("BULLET:") else "reconstruct"
        item = "slow" if "slow" in prompt else "bad" if "bad" in prompt else "fast"
        self.events.append(
            {
                "event": "start",
                "item": item,
                "step": step,
                "time": time.monotonic(),
                "in_flight": self.current_in_flight,
            }
        )

        if item == "slow" and step == "bullet":
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

        if item == "bad" and step == "bullet":
            content = ""
        elif step == "bullet":
            content = f"bullet:{item}"
        else:
            content = f"reconstructed {item}"

        return httpx.Response(
            200,
            request=httpx.Request("POST", "http://fake-server/v1/chat/completions"),
            content=json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8"),
        )


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


class CPTPipelinePoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_candidate_completes_without_waiting_for_slow_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "source.jsonl"
            write_jsonl(
                source_path,
                [
                    {"id": "slow", "title": "Slow", "content": "slow"},
                    {"id": "fast", "title": "Fast", "content": "fast"},
                ],
            )
            pipeline = FakeCPTPipelinePool(tmp_path, max_in_flight=2)
            try:
                await pipeline.build_dataset(source_path)
            finally:
                await pipeline.aclose()

            all_records = [
                json.loads(line)
                for line in (tmp_path / "output" / "all.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        slow_bullet_end = next(
            event["time"]
            for event in pipeline.events
            if event["event"] == "end" and event["item"] == "slow" and event["step"] == "bullet"
        )
        fast_reconstruct_end = next(
            event["time"]
            for event in pipeline.events
            if event["event"] == "end" and event["item"] == "fast" and event["step"] == "reconstruct"
        )
        fast_steps = [
            event["step"]
            for event in pipeline.events
            if event["event"] == "start" and event["item"] == "fast"
        ]

        self.assertLess(fast_reconstruct_end, slow_bullet_end)
        self.assertEqual(fast_steps, ["bullet", "reconstruct"])
        self.assertLessEqual(pipeline.max_observed_in_flight, 2)
        self.assertEqual({record["id"] for record in all_records}, {"slow", "fast"})
        self.assertTrue(all(record["copyright_mitigation"] for record in all_records))
        self.assertEqual({record["cpt_generator"] for record in all_records}, {"fake-cpt-model"})

    async def test_one_failed_candidate_does_not_stop_pipeline_and_preserves_chunk_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "source.jsonl"
            write_jsonl(
                source_path,
                [
                    {"id": "bad", "title": "Bad", "content": "bad"},
                    {"id": "fast", "title": "Fast", "content": "fast"},
                ],
            )
            pipeline = FakeCPTPipelinePool(tmp_path, max_in_flight=2)
            try:
                await pipeline.build_dataset(source_path)
            finally:
                await pipeline.aclose()

            output_dir = tmp_path / "output"
            all_records = [
                json.loads(line)
                for line in (output_dir / "all.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            failure = json.loads((output_dir / "all.failures.jsonl").read_text(encoding="utf-8").strip())
            cached_ids = (output_dir / "cache_processed_ids.txt").read_text(encoding="utf-8").splitlines()

        self.assertEqual([record["id"] for record in all_records], ["fast"])
        self.assertEqual(failure["id"], "bad")
        self.assertEqual(failure["chunk_index"], 0)
        self.assertEqual(failure["failed_step"], "to_bullet_points")
        self.assertIn("bad", failure["original_text"])
        self.assertEqual(cached_ids, ["fast"])
        self.assertLessEqual(pipeline.max_observed_in_flight, 2)

    async def test_without_copyright_mitigation_no_http_request_is_sent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "source.jsonl"
            write_jsonl(source_path, [{"id": "plain", "title": "Plain", "content": "plain text"}])
            pipeline = FakeCPTPipelinePool(tmp_path, max_in_flight=1, copyright_mitigation=False)
            try:
                await pipeline.build_dataset(source_path)
            finally:
                await pipeline.aclose()

            all_records = [
                json.loads(line)
                for line in (tmp_path / "output" / "all.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(pipeline.payloads, [])
        self.assertEqual(all_records[0]["id"], "plain")
        self.assertEqual(all_records[0]["text"], "plain text")
        self.assertFalse(all_records[0]["copyright_mitigation"])
        self.assertNotIn("cpt_generator", all_records[0])

    async def test_original_fallback_saves_failed_mitigation_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "source.jsonl"
            write_jsonl(
                source_path,
                [
                    {"id": "bad", "title": "Bad", "content": "bad"},
                    {"id": "fast", "title": "Fast", "content": "fast"},
                ],
            )
            pipeline = FakeCPTPipelinePool(
                tmp_path,
                max_in_flight=2,
                copyright_mitigation_failure_policy="original",
            )
            try:
                await pipeline.build_dataset(source_path)
            finally:
                await pipeline.aclose()

            output_dir = tmp_path / "output"
            all_records = [
                json.loads(line)
                for line in (output_dir / "all.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            failures_path = output_dir / "all.failures.jsonl"
            cached_ids = (output_dir / "cache_processed_ids.txt").read_text(encoding="utf-8").splitlines()

        records_by_id = {record["id"]: record for record in all_records}
        self.assertEqual(set(records_by_id), {"bad", "fast"})
        self.assertEqual(records_by_id["bad"]["text"], "bad")
        self.assertFalse(records_by_id["bad"]["copyright_mitigation"])
        self.assertNotIn("cpt_generator", records_by_id["bad"])
        self.assertFalse(failures_path.exists())
        self.assertEqual(set(cached_ids), {"bad", "fast"})

    async def test_cpt_enable_thinking_false_is_sent_in_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "source.jsonl"
            write_jsonl(source_path, [{"id": "fast", "title": "Fast", "content": "fast"}])
            pipeline = FakeCPTPipelinePool(tmp_path, max_in_flight=1, cpt_enable_thinking=False)
            try:
                await pipeline.build_dataset(source_path)
            finally:
                await pipeline.aclose()

        self.assertTrue(pipeline.payloads)
        self.assertTrue(
            all(
                payload["chat_template_kwargs"]["enable_thinking"] is False
                for payload in pipeline.payloads
            )
        )

    async def test_cpt_enable_thinking_true_is_sent_in_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "source.jsonl"
            write_jsonl(source_path, [{"id": "fast", "title": "Fast", "content": "fast"}])
            pipeline = FakeCPTPipelinePool(tmp_path, max_in_flight=1, cpt_enable_thinking=True)
            try:
                await pipeline.build_dataset(source_path)
            finally:
                await pipeline.aclose()

        self.assertTrue(pipeline.payloads)
        self.assertTrue(
            all(
                payload["chat_template_kwargs"]["enable_thinking"] is True
                for payload in pipeline.payloads
            )
        )


if __name__ == "__main__":
    unittest.main()
