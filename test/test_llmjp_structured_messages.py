"""Offline regression coverage for the custom training schema and migration."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from migrate_reasoning_effort_output import migrate
from reasoning_effort.formatters import LLMJP4ReasoningEffortFormatter, Qwen38ReasoningEffortFormatter
from reasoning_effort.output_schema import compact_output, validate_llmjp_messages
from test.test_reasoning_effort_dataset import FakeTokenizer, THINKING, ROW, pipeline, settings


def canonical(effort):
    return dict(ROW, thinking=THINKING, question='  質問\n', answer=' 答え\n',
                canonical_record_id='sample-' + effort, canonical_reasoning_effort=effort,
                source_qa_id='source-1', source_metadata={'qa_id': 'source-1', 'id': None, 'chunk_index': 2},
                thinking_generator='fake')


def expected_messages(row, effort):
    return [{'role': 'user', 'content': row['question']},
            {'role': 'assistant', 'content': row['answer'], 'thinking': row['thinking']}]


class StructuredTests(unittest.TestCase):
    def test_all_efforts_and_unchanged_export_values(self):
        for effort, system_effort, qwen_effort in [('low', 'Low', 'low'), ('medium', 'Medium', 'medium'), ('high', 'High', 'xhigh')]:
            with self.subTest(effort=effort):
                source = canonical(effort)
                for family, formatter, mapped, multiplier in [
                    ('llm_jp_4', LLMJP4ReasoningEffortFormatter, effort, 2),
                    ('qwen3_8', Qwen38ReasoningEffortFormatter, qwen_effort, 1),
                ]:
                    tokenizer = FakeTokenizer(multiplier)
                    f = formatter(tokenizer, 'test')
                    row = compact_output(f.format(source))
                    # Full pre-change export contract, excluding only LLM-jp messages.
                    expected = {
                        'qa_id': source['canonical_record_id'] + ':' + family,
                        'question': source['question'], 'answer': source['answer'], 'thinking': THINKING,
                        'reasoning_effort': mapped, 'eval': '5',
                        'messages': [{'role': 'user', 'content': source['question']},
                                     {'role': 'assistant', 'content': source['answer'],
                                      'reasoning_content' if family == 'qwen3_8' else 'thinking': THINKING}],
                        'thinking_tokens': len(THINKING.split()) * multiplier,
                        'thinking_generator': 'fake', 'source_metadata': source['source_metadata'],
                    }
                    if family == 'llm_jp_4':
                        expected['chat_template_kwargs'] = {'reasoning_effort': effort, 'conversation_start_date': '2026-09-11'}
                        self.assertEqual(row['messages'], expected_messages(source, system_effort))
                        self.assertEqual(len(tokenizer.calls), 1)
                        row.pop('messages')
                        expected.pop('messages')
                    else:
                        expected['chat_template_kwargs'] = {'reasoning_effort': mapped, 'enable_thinking': True, 'preserve_thinking': True}
                        self.assertEqual(len(tokenizer.calls), 1)
                    self.assertEqual(row, expected)
                    with self.assertRaises(ValueError):
                        f.format(dict(source, thinking='broken'))
                with self.assertRaises(ValueError):
                    LLMJP4ReasoningEffortFormatter(FakeTokenizer(), 'test').format(
                        dict(source, canonical_reasoning_effort='unknown'))

    def test_reject_structural_corruption(self):
        row = LLMJP4ReasoningEffortFormatter(FakeTokenizer(), 'test').format(canonical('low'))
        mutations = [
            lambda m: m.pop(), lambda m: m.reverse(),
            lambda m: m[0].update(channel='system'),
            lambda m: m[1].update(channel='analysis'),
            lambda m: m[1].update(name='user'),
            lambda m: m[1].update(content='question'),
            lambda m: m[1].update(thinking=' ' + THINKING),
            lambda m: m[1].update(reasoning_content=THINKING),
        ]
        for mutate in mutations:
            bad = deepcopy(row)
            mutate(bad['messages'])
            with self.assertRaises(ValueError):
                compact_output(bad)

    def test_messages_only_migration_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for family, formatter in [('llm_jp_4', LLMJP4ReasoningEffortFormatter), ('qwen3_8', Qwen38ReasoningEffortFormatter)]:
                rows = [formatter(FakeTokenizer(), 'test').format(canonical(e)) for e in ('low', 'medium', 'high')]
                for row in rows:
                    row['unknown_metadata'] = {'nested': [None, True, 123]}
                    if family == 'llm_jp_4':
                        row['messages'] = [{'role': 'user', 'content': row['question']},
                                           {'role': 'assistant', 'content': row['answer'], 'thinking': row['thinking']}]
                source, dest, again = (root / (family + suffix) for suffix in ('.jsonl', '-new.jsonl', '-again.jsonl'))
                source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                before = source.read_bytes()
                self.assertEqual(migrate(source, dest, messages_only=True), 3)
                actual = [json.loads(line) for line in dest.read_text().splitlines()]
                for old, new in zip(rows, actual):
                    if family == 'llm_jp_4':
                        validate_llmjp_messages(new)
                        self.assertEqual({k: v for k, v in old.items() if k != 'messages'},
                                         {k: v for k, v in new.items() if k != 'messages'})
                    else:
                        self.assertEqual(old, new)
                migrate(dest, again, messages_only=True)
                self.assertEqual(dest.read_bytes(), again.read_bytes())
                for collision in (source, dest):
                    with self.assertRaises(ValueError):
                        migrate(source, collision, messages_only=True)
                self.assertEqual(source.read_bytes(), before)
                self.assertEqual(dest.read_bytes(), again.read_bytes())
                blocked = root / (family + '-blocked.jsonl')
                blocked_sidecar = Path(str(blocked) + '.resume.jsonl')
                blocked_sidecar.write_text('existing resume')
                with self.assertRaises(ValueError):
                    migrate(source, blocked, messages_only=True)
                self.assertEqual(blocked_sidecar.read_text(), 'existing resume')
                self.assertFalse(blocked.exists())
            for bad in ('{broken\n', json.dumps(dict(rows[0], qa_id='bad:llm_jp_4', reasoning_effort='unknown')) + '\n'):
                source = root / 'bad.jsonl'
                source.write_text(bad)
                before = source.read_bytes()
                with self.assertRaises(ValueError):
                    migrate(source, root / 'bad-new.jsonl', messages_only=True)
                self.assertEqual(source.read_bytes(), before)
                self.assertFalse((root / 'bad-new.jsonl').exists())
                self.assertFalse((root / 'bad-new.jsonl.resume.jsonl').exists())


class ResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_migrated_compact_output_resumes_without_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = settings(directory)
            p = pipeline(cfg)
            try:
                await p.run([ROW])
            finally:
                await p.aclose()
            source = Path(cfg['output']['llm_jp_4']['path'])
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            metadata = [json.loads(line) for line in Path(str(source) + '.resume.jsonl').read_text().splitlines()]
            for row, meta in zip(rows, metadata):
                row['qa_id'] = meta['canonical_record_id'] + ':llm_jp_4'
                row['messages'] = [{'role': 'user', 'content': row['question']},
                                   {'role': 'assistant', 'content': row['answer'], 'thinking': row['thinking']}]
            source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            old_bytes = source.read_bytes()
            old_resume = pipeline(cfg)
            try:
                await old_resume.run([ROW])
                self.assertEqual(old_resume.generator.calls, [])
                self.assertEqual(source.read_bytes(), old_bytes)
            finally:
                await old_resume.aclose()
            dest = source.with_name('structured.jsonl')
            migrate(source, dest, messages_only=True)
            cfg['output']['llm_jp_4']['path'] = str(dest)
            before = dest.read_bytes()
            resumed = pipeline(cfg)
            try:
                await resumed.run([ROW])
                self.assertEqual(resumed.generator.calls, [])
                self.assertEqual(resumed.stats['llm_jp_4_skipped'], 3)
                self.assertEqual(dest.read_bytes(), before)
            finally:
                await resumed.aclose()

    async def test_unknown_effort_recorded_by_existing_failure_path(self):
        with tempfile.TemporaryDirectory() as directory:
            p = pipeline(settings(directory))
            original = p.formatters['llm_jp_4'].format
            try:
                with patch.object(p.formatters['llm_jp_4'], 'format',
                                  side_effect=lambda row: original(dict(row, canonical_reasoning_effort='unknown'))):
                    await p.run([ROW])
                self.assertEqual(p.stats['llm_jp_4_formatting_terminal_failures'], 3)
                self.assertEqual(len(p.generator.calls), 3)
                self.assertEqual(len(p.outputs['llm_jp_4'].records), 0)
                failures = Path(p.settings['output']['failures_path']).read_text()
                self.assertIn('Unsupported canonical effort', failures)
            finally:
                await p.aclose()
