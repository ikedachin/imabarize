import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from commons.output_validation import OutputValidator
from main_create_imabari_qa import process_json_files, process_text_files
from test.test_create_qa_model import FakePipelinePool


PROFILE_FILE = Path(__file__).resolve().parents[1] / "yamls/qa_output_validation_format.yaml"
VALID = "## 思考プロセス\n\n### 1. 整理\n\n本文です。"


def config():
    return {
        "profiles_file": str(PROFILE_FILE),
        "fields": {"thinking": {"profile": "thinking_markdown_v2"}},
    }


class FormatTests(unittest.TestCase):
    def setUp(self):
        self.validator = OutputValidator(config())

    def test_valid_and_section_limit(self):
        for count in (1, 5, 6):
            value = "## 思考プロセス\n\n" + "\n\n".join(
                f"### {i}. 整理\n\n本文です。" for i in range(1, count + 1)
            )
            with self.subTest(count=count):
                errors = self.validator.validate({"thinking": value})
                self.assertEqual(bool(errors), count == 6)
        self.assertFalse(self.validator.validate({"thinking": "\n" + VALID + "\n\n"}))

    def test_every_heading_requires_actual_lf_blank_line(self):
        for value in (
            VALID.replace("プロセス\n\n", "プロセス\n"),
            VALID.replace("整理\n\n", "整理\n"),
            VALID.replace("\n\n", "\n \n"),
            VALID.replace("\n\n", "\n\t\n"),
            VALID.replace("\n", "\r\n"),
            VALID + "\n### 2. 末尾\n",
            VALID + "\n### 2. 末尾\n \n本文",
        ):
            with self.subTest(value=value):
                errors = self.validator.validate({"thinking": value})
                self.assertTrue(any(e["rule"] == "blank_line_after" for e in errors))
        self.assertFalse(self.validator.validate({"thinking": VALID.replace("\n\n", "\n\n\n")}))

    def test_title_and_second_nonblank_line(self):
        for value in (None, "", "説明\n" + VALID, VALID.replace("### 1.", "### 整理"),
                      VALID.replace("\n\n###", "\n\n前置き\n###")):
            with self.subTest(value=value):
                self.assertTrue(self.validator.validate({"thinking": value}))

    def test_forbidden_suffix_and_normal_body(self):
        for suffix in ('```', "'''", '"', '"""', "文字数：350文字", "（350文字）", "---", "!!!"):
            with self.subTest(suffix=suffix):
                errors = self.validator.validate({"thinking": VALID + "\n" + suffix + "\n\n"})
                self.assertTrue(any(e["rule"] == "forbidden_suffix_lines" for e in errors))
        errors = self.validator.validate({"thinking": VALID + '\n```\n文字数：350文字\n"'})
        self.assertEqual(len(errors), 3)
        for body in ('「引用」です。', '350文字の説明を確認する。', '本文は "引用" を含みます。'):
            self.assertFalse(self.validator.validate({"thinking": VALID + "\n" + body}))

    def test_fields_share_profile_and_input_is_unchanged(self):
        settings = config()
        settings["fields"]["answer"] = {"profile": "thinking_markdown_v2"}
        validator = OutputValidator(settings)
        values = {"thinking": VALID, "answer": "bad"}
        original = dict(values)
        self.assertEqual({e["field"] for e in validator.validate(values)}, {"answer"})
        self.assertEqual(values, original)
        self.assertEqual(OutputValidator().validate({}), [])
        self.assertEqual(OutputValidator({"enabled": False}).validate({}), [])

    def test_bad_settings_fail_at_initialization(self):
        profiles = yaml.safe_load(PROFILE_FILE.read_text())
        base = {"profiles": profiles, "fields": config()["fields"]}
        variants = []
        for rule in (
            {"type": "unknown"},
            {"type": "line_fullmatch", "index": 0, "pattern": "["},
            {"type": "matching_line_count", "pattern": ".*", "min": 5, "max": 1},
        ):
            variant = copy.deepcopy(base)
            variant["profiles"]["thinking_markdown_v2"]["rules"] = [rule]
            variants.append(variant)
        variants.append({"fields": {"thinking": {"profile": "missing"}}})
        variants.append({"on_failure": "ignore"})
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                OutputValidator(variant)


class ValidatedPipeline(FakePipelinePool):
    def __init__(self, path, initial=VALID, refined=None):
        super().__init__(path)
        self.output_validator = OutputValidator(config())
        self.initial = initial
        self.refined = refined

    async def _infer_text_async(self, prompt, step=None):
        if step == 3:
            return f"<think>{self.initial}</think>"
        if step == 4:
            thinking = f"<think>{self.refined}</think>" if self.refined is not None else ""
            return thinking + "<answer>修正回答</answer>"
        return await super()._infer_text_async(prompt, step=step)


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_final_values_checked_before_eval(self):
        for initial, refined, failed in ((VALID, None, False), (VALID, "bad", True),
                                         ("bad", VALID, False), ("bad", None, True)):
            with self.subTest(initial=initial, refined=refined), tempfile.TemporaryDirectory() as tmp:
                pipeline = ValidatedPipeline(Path(tmp), initial, refined)
                try:
                    result = (await pipeline.create_qa_batch_async(["fast"], 1))[0]
                finally:
                    await pipeline.aclose()
                self.assertEqual(bool(result.get("failed")), failed)
                self.assertEqual(any(e["step"] == "eval" for e in pipeline.events), not failed)
                if failed:
                    self.assertEqual(result["error_type"], "output_format")
                    self.assertTrue(result["validation_errors"])
                    self.assertEqual(result["previous_outputs"]["thinking"], refined or initial)
                else:
                    self.assertEqual(result["thinking"], refined or initial)
                    self.assertIn(result["thinking"], result["messages"][1]["content"])

    async def test_rejected_json_saved_without_cache_or_normal_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            source.write_text(json.dumps({"id": "source-id", "chunk_index": 2, "text": "fast"}) + "\n")
            pipeline = ValidatedPipeline(root, refined=VALID + '\n"')
            try:
                await process_json_files(pipeline, [source], "text", 1, 0)
            finally:
                await pipeline.aclose()
            failure = json.loads((pipeline.output_dir / "source.failures.jsonl").read_text())
            self.assertEqual(failure["id"], "source-id")
            self.assertEqual(failure["chunk_index"], 2)
            self.assertIn("qa_id", failure)
            self.assertNotIn("item_id", failure)
            self.assertTrue(failure["previous_outputs"]["thinking"].endswith('"'))
            self.assertFalse((pipeline.output_dir / "source.jsonl").exists())
            self.assertFalse(list(pipeline.output_dir.glob("cache_*.txt")))

    async def test_text_failure_uses_same_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.txt"
            source.write_text("fast")
            pipeline = ValidatedPipeline(root, initial="bad")
            try:
                await process_text_files(pipeline, [source], 1, 0)
            finally:
                await pipeline.aclose()
            failure = json.loads(next(pipeline.output_dir.glob("*.failures.jsonl")).read_text())
            self.assertEqual(failure["error_type"], "output_format")
            self.assertEqual(failure["source_files"], [source.name])


if __name__ == "__main__":
    unittest.main()
