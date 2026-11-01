import argparse
import asyncio
import hashlib
import json
import random
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from commons.util_settings import load_settings
from commons.utils_msg import msg_debug, msg_error, msg_info, msg_success
from main_create_imabari_qa import collect_source_files, get_parent_book_name, load_json_entries
from pipelines.create_qa_model import PipelineJob, QAPipeline


class EvalQAPipeline(QAPipeline):
    """Generate only a question and an answer, without a thinking step."""

    def __init__(self, settings: Dict[str, Any]):
        settings = dict(settings)
        settings["prompts"] = [
            prompt for prompt in settings.get("prompts", [])
            if "question_prompt" in prompt or "answer_prompt" in prompt
        ]
        super().__init__(settings)

    async def _run_one_step(self, job: PipelineJob) -> PipelineJob:
        if job.step not in (1, 2):
            return PipelineJob(job.item_id, job.step + 1, job.payload, job.previous_outputs)
        tag = "question" if job.step == 1 else "answer"
        template = self.prompts.get(f"{tag}_prompt")
        if not template:
            raise ValueError(f"{tag}_prompt must be set in prompts.")
        raw = await self._infer_text_async(
            template.format(
                text=job.payload.get("text", ""),
                question=job.previous_outputs.get("question", ""),
                random_token=job.payload.get("random_token", ""),
            ),
            step=job.step,
        )
        start, end = f"<{tag}>", f"</{tag}>"
        if raw.count(start) != 1 or raw.count(end) != 1 or not raw.strip().startswith(start) or not raw.strip().endswith(end):
            raise ValueError(f"Step {job.step} did not return a single {tag} tag.")
        value = self._extract_tag(raw, tag)
        if not value or value == "作成不可" or "<think>" in value or "</think>" in value:
            raise ValueError(f"Step {job.step} returned an unusable {tag}.")
        outputs = dict(job.previous_outputs)
        outputs[tag] = value
        return PipelineJob(job.item_id, job.step + 1, job.payload, outputs)

    def _final_result(self, item_id: int, outputs: Dict[str, str]) -> Dict[str, Any]:
        question, answer = outputs["question"], outputs["answer"]
        return {
            "item_id": item_id,
            "qa_id": str(uuid.uuid4()),
            "question": question,
            "answer": answer,
            "qa_generator": self.inference_config.get("MODEL_NAME", ""),
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }


def parse_sample_size(value: Any) -> Optional[int]:
    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def select_sampled_items(items: List[Dict[str, Any]], sample_size: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if sample_size is None or len(items) <= sample_size:
        return list(items)
    return random.Random(seed).sample(items, sample_size)


def collect_candidates(text_files: List[Path], json_files: List[Path], target_key: Optional[str], output_dir: Path) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for path in sorted(text_files + json_files):
        if path in text_files:
            try:
                text = path.read_text(encoding="utf-8")
            except Exception as exc:
                print(msg_error(f"Failed to read {path}: {exc}"))
                continue
            entries = [(0, {"text": text, "id": str(uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())))})]
            output_jsonl = output_dir / f"{get_parent_book_name(path)}.jsonl"
        else:
            entries = list(enumerate(load_json_entries(path)))
            output_jsonl = output_dir / f"{path.stem}.jsonl"
        for row_index, entry in entries:
            value = entry.get("text" if path in text_files else target_key)
            if not isinstance(value, str) or not value.strip():
                print(msg_debug(f"Skipping empty or missing source text: {path.name}, row={row_index}"))
                continue
            # Identify source rows independently of IDs, which need not be unique.
            identity = json.dumps([str(path.resolve()), row_index, value], ensure_ascii=False)
            candidates.append({
                "text": value,
                "source_file": path.name,
                "output_jsonl": output_jsonl,
                "entry_id": str(entry.get("id") or f"{path.stem}:{row_index}"),
                "chunk_index": entry.get("chunk_index"),
                "cache_key": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            })
    return candidates


async def main(settings_path: str | None, source_path: str | None, target_key: str | None, start_index: int, sample_size_override: Optional[int] = None) -> None:
    if settings_path is None or source_path is None:
        print(msg_error("settings_path and source path are required."), file=sys.stderr)
        sys.exit(1)
    settings = load_settings(Path(settings_path))
    source = Path(source_path).expanduser().resolve()
    text_files, json_files = collect_source_files(source)
    if not text_files and not json_files:
        print(msg_error(f"No supported files were found for {source}."), file=sys.stderr)
        sys.exit(1)
    if json_files and not target_key:
        print(msg_error("target_key is required when processing JSON files."), file=sys.stderr)
        sys.exit(1)

    sample_size = parse_sample_size(sample_size_override if sample_size_override is not None else settings.get("sample_size", 300))
    seed = int(settings.get("seed", 42))
    pipeline = EvalQAPipeline(settings)
    try:
        candidates = collect_candidates(text_files, json_files, target_key, pipeline.output_dir)
        selected = select_sampled_items(candidates, sample_size, seed)
        # Apply resume after sampling so repeated runs always address the same sample.
        if start_index > 1:
            selected = selected[start_index - 1:]
        cache_file = pipeline.output_dir / "cache_eval_qa.txt"
        processed = set(cache_file.read_text(encoding="utf-8").splitlines()) if cache_file.exists() else set()
        pending = [item for item in selected if item["cache_key"] not in processed]
        print(msg_info(f"Evaluation sources={len(candidates)}, selected={len(selected)}, pending={len(pending)}, seed={seed}"))
        batch_size = int(settings.get("pipeline_batch_size", max(int(settings.get("batch_size", 1)), pipeline.max_in_flight * 4)))

        async def on_result(item_id: int, result: Dict[str, Any]) -> None:
            candidate = pending[item_id]
            result.update(source_files=[candidate["source_file"]], id=candidate["entry_id"], chunk_index=candidate["chunk_index"])
            if result.get("failed"):
                failure_path = candidate["output_jsonl"].with_suffix(".failures.jsonl")
                pipeline.append_failure_jsonl(failure_path, result)
                print(msg_error(f"Failed to create eval QA; saved to {failure_path}"))
                return
            pipeline.append_jsonl(candidate["output_jsonl"], result)
            with cache_file.open("a", encoding="utf-8") as f:
                f.write(candidate["cache_key"] + "\n")

        await pipeline.create_qa_batch_async([item["text"] for item in pending], batch_size=batch_size, on_result=on_result)
    finally:
        await pipeline.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a seeded evaluation Q&A sample from text, markdown, JSON, and JSONL.")
    parser.add_argument("-p", "--settings_path", default="./yamls/create_eval_qa_settings_format.yaml")
    parser.add_argument("-s", "--source", required=True, help="Source file or directory")
    parser.add_argument("-t", "--target_key", default=None, help="Text field in JSON/JSONL sources")
    parser.add_argument("-i", "--start_index", type=int, default=0, help="1-based starting position in the selected sample")
    parser.add_argument("-n", "--sample_size", type=int, default=None, help="Override YAML sample_size; 0 or less selects all")
    args = parser.parse_args()
    asyncio.run(main(args.settings_path, args.source, args.target_key, args.start_index, args.sample_size))
    print(msg_success("Evaluation Q&A Creation Pipeline Completed"))
