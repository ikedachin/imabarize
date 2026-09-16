"""Legacy IDs, directory migration and strict new-format validation."""
import json
from pathlib import Path
import tempfile
import unittest

from migrate_reasoning_effort_output import migrate
from migrate_reasoning_effort_directory import build, CACHE, OUTPUTS
from reasoning_effort.cache import stable_id
from reasoning_effort.output_schema import compact_output
from test.test_reasoning_effort_dataset import ROW, pipeline, settings


def write(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class ModelOutputMigrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_id_recovery_directory_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = settings(tmp)
            cfg['output']['cache_path'] = str(root / CACHE)
            for family, name in zip(('qwen3_8', 'llm_jp_4'), OUTPUTS):
                cfg['output'][family]['path'] = str(root / name)
            p = pipeline(cfg)
            try:
                await p.run([ROW])
            finally:
                await p.aclose()
            expected = {}
            for family in ('qwen3_8', 'llm_jp_4'):
                source = Path(cfg['output'][family]['path'])
                expected[family] = read(source)
                rows = read(source)
                sidecar = Path(str(source) + '.resume.jsonl')
                metadata = read(sidecar)
                for row, meta in zip(rows, metadata):
                    row.pop('qa_id')
                    row['source_qa_id'] = ROW['qa_id']
                    if family == 'qwen3_8':
                        row.pop('chat_template_kwargs')
                    else:
                        row['messages'] = [
                            {'role': 'system', 'content': [{'type': 'system_content'}]},
                            {'role': 'user', 'content': [{'type': 'text', 'text': row['question']}]},
                            {'role': 'assistant', 'channel': 'analysis_imabari',
                             'content': [{'type': 'text', 'text': row['thinking']}]},
                            {'role': 'assistant', 'channel': 'final_imabari',
                             'content': [{'type': 'text', 'text': row['answer']}]},
                        ]
                    meta['output_digest'] = stable_id(row)
                write(source, rows)
                write(sidecar, metadata)
                dest = root / ('new-' + source.name)
                migrate(source, dest)
                self.assertEqual(read(dest), expected[family])
                again = root / ('again-' + source.name)
                migrate(dest, again)
                self.assertEqual(dest.read_bytes(), again.read_bytes())
                cfg['output'][family]['path'] = str(dest)
                # A missing digest mapping must not manufacture an ID.
                write(sidecar, [])
                with self.assertRaisesRegex(ValueError, 'Missing or ambiguous'):
                    migrate(source, root / ('bad-' + source.name))
                write(sidecar, metadata)
            destination, stage = root / 'destination', root / 'stage'
            destination.mkdir()
            stage.mkdir()
            summary = build(root, destination, stage)
            for family, name in zip(('qwen3_8', 'llm_jp_4'), OUTPUTS):
                self.assertEqual(summary[name], 3)
                self.assertEqual(read(stage / name), expected[family])
            resumed = pipeline(cfg)
            try:
                await resumed.run([ROW])
                self.assertEqual(resumed.generator.calls, [])
                for family in expected:
                    self.assertEqual(resumed.stats[family + '_skipped'], 3)
            finally:
                await resumed.aclose()

    async def test_invalid_flags_and_family_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = settings(tmp)
            p = pipeline(cfg)
            try:
                await p.run([ROW])
                q = next(iter(p.outputs['qwen3_8'].records.values()))
                self.assertTrue(q['qa_id'].endswith(':qwen3_8'))
                for flag in ('enable_thinking', 'preserve_thinking'):
                    for value in (False, 1, None):
                        with self.assertRaises(ValueError):
                            compact_output(dict(q, chat_template_kwargs=dict(q['chat_template_kwargs'], **{flag: value})))
                with self.assertRaises(ValueError):
                    compact_output(dict(q, qa_id=q['qa_id'].replace(':qwen3_8', ':llm_jp_4')))
            finally:
                await p.aclose()
            cfg['output']['qwen3_8']['path'], cfg['output']['llm_jp_4']['path'] = (
                cfg['output']['llm_jp_4']['path'], cfg['output']['qwen3_8']['path'])
            with self.assertRaisesRegex(ValueError, 'model family mismatch'):
                pipeline(cfg)
