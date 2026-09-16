import json
from pathlib import Path
import tempfile
import unittest

from migrate_reasoning_effort_output import migrate
from reasoning_effort.output_schema import OUTPUT_KEYS, compact_output
from test.test_reasoning_effort_dataset import ROW, pipeline, settings


class CompactOutputTests(unittest.IsolatedAsyncioTestCase):
    async def test_effort_mapping_and_message_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            p = pipeline(settings(directory))
            try:
                await p.run([ROW])
                for key, canonical in p.cache.records.items():
                    q = p.outputs['qwen3_8'].records[key]
                    j = p.outputs['llm_jp_4'].records[key]
                    label = canonical['canonical_reasoning_effort']
                    self.assertEqual(q['reasoning_effort'], {'low': 'low', 'medium': 'medium', 'high': 'xhigh'}[label])
                    self.assertEqual(j['reasoning_effort'], label)
                    with self.assertRaises(ValueError):
                        compact_output(dict(q, reasoning_effort='high'))
                    with self.assertRaises(ValueError):
                        compact_output(dict(j, messages=[{'role': 'assistant', 'channel': 'analysis', 'content': j['thinking']}]))
                    for record in (q, j):
                        altered = dict(record, messages=[{'role': 'user', 'content': 'wrong'}, record['messages'][1]])
                        with self.assertRaises(ValueError):
                            compact_output(altered)
                    with self.assertRaises(ValueError):
                        p.formatters['qwen3_8'].format(dict(canonical, canonical_reasoning_effort='xhigh'))
            finally:
                await p.aclose()

    async def test_compact_resume_and_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = settings(directory)
            row = dict(ROW, id=123, unknown='do not export')
            p = pipeline(cfg)
            try:
                await p.run([row])
                self.assertEqual(len(p.generator.calls), 3)
                legacy = {}
                for family, journal in p.outputs.items():
                    legacy[family] = []
                    for key, record in journal.records.items():
                        original = p.formatters[family].format(p.cache.records[key])
                        legacy[family].append(original)
                        self.assertEqual(list(record), list(OUTPUT_KEYS))
                        self.assertEqual(record['source_metadata'], {'qa_id': ROW['qa_id'], 'id': 123, 'chunk_index': 2})
                        self.assertNotIn('original_thinking', record)
                        for field in ('messages', 'question', 'answer', 'thinking'):
                            self.assertEqual(record[field], original[field])
                        self.assertNotIn('text', record)
                        self.assertNotIn('template_messages', record)
                        if family == 'llm_jp_4':
                            self.assertEqual(record['chat_template_kwargs'], original['chat_template_kwargs'])
                            # A previous on-disk schema used channel messages and an adapter.
                            legacy[family][-1] = dict(original,
                                template_messages=original['messages'], text='legacy rendered text',
                                messages=[{'role': 'assistant', 'channel': 'analysis',
                                           'content': [{'type': 'text', 'text': original['thinking']}]}])
                        else:
                            self.assertEqual(record['chat_template_kwargs'], {'reasoning_effort': record['reasoning_effort'], 'enable_thinking': True, 'preserve_thinking': True})
                            self.assertIn(record['reasoning_effort'], ('low', 'medium', 'xhigh'))
                paths = {f: j.path for f, j in p.outputs.items()}
            finally:
                await p.aclose()
            before = {f: path.read_bytes() for f, path in paths.items()}
            resumed = pipeline(cfg)
            try:
                await resumed.run([row])
                self.assertEqual(resumed.generator.calls, [])
                self.assertEqual(before, {f: path.read_bytes() for f, path in paths.items()})
            finally:
                await resumed.aclose()
            # Legacy rows mixed with new rows must remain untouched on resume.
            for family, path in paths.items():
                records = [legacy[family][0]] + [json.loads(l) for l in path.read_text().splitlines()][1:]
                path.write_text(''.join(json.dumps(r) + '\n' for r in records))
            mixed = pipeline(cfg)
            try:
                await mixed.run([row])
                self.assertEqual(mixed.generator.calls, [])
                self.assertTrue(all(mixed.stats[f + '_skipped'] == 3 for f in paths))
            finally:
                await mixed.aclose()
            for family, path in paths.items():
                dest = path.with_name('compact-' + path.name)
                self.assertEqual(migrate(path, dest), 3)
                actual = [json.loads(l) for l in dest.read_text().splitlines()]
                self.assertTrue(all('original_thinking' not in r for r in actual))
                expected = [compact_output(dict(r, messages=r['template_messages']))
                            if 'template_messages' in r else compact_output(r)
                            for r in legacy[family]]
                self.assertEqual(actual, expected)
                with self.assertRaises(ValueError):
                    migrate(path, dest)
                # The converted output, with its sidecar, can itself be converted.
                self.assertEqual(migrate(dest, path.with_name('again-' + path.name)), 3)
            invalid = Path(directory) / 'invalid.jsonl'
            invalid_dest = Path(directory) / 'invalid-out.jsonl'
            q = legacy['qwen3_8'][0]
            j = legacy['llm_jp_4'][0]
            for records in ([q, q], [q, j], [dict(q, reasoning_effort='high')]):
                invalid.write_text(''.join(json.dumps(r) + '\n' for r in records))
                with self.assertRaises(ValueError):
                    migrate(invalid, invalid_dest)
                self.assertFalse(invalid_dest.exists())
                self.assertFalse(Path(str(invalid_dest) + '.resume.jsonl').exists())
            # Tokenizer metadata continues to protect compact records.
            changed = pipeline(cfg)
            changed.formatters['qwen3_8'].revision = 'different'
            try:
                await changed.run([row])
                self.assertEqual(changed.generator.calls, [])
                self.assertEqual(changed.stats['qwen3_8_formatting_terminal_failures'], 3)
            finally:
                await changed.aclose()

    async def test_missing_optional_and_disabled_original(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = settings(directory)
            cfg['output']['keep_original_thinking'] = False
            row = {k: v for k, v in ROW.items() if k not in ('eval', 'chunk_index', 'qa_id')}
            p = pipeline(cfg)
            try:
                await p.run([row])
                for journal in p.outputs.values():
                    for record in journal.records.values():
                        self.assertNotIn('original_thinking', record)
                        self.assertIsNone(record['eval'])
                        for field in ('source_qa_id', 'id', 'chunk_index'):
                            self.assertNotIn(field, record)
                        self.assertEqual(record['source_metadata'], dict(qa_id=None, id=None, chunk_index=None))
            finally:
                await p.aclose()

    async def test_identifier_removal_migration_and_resume_both_models(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = settings(directory)
            p = pipeline(cfg)
            try:
                await p.run([ROW])
            finally:
                await p.aclose()
            for family in ('qwen3_8', 'llm_jp_4'):
                source = Path(cfg['output'][family]['path'])
                original = [json.loads(line) for line in source.read_text().splitlines()]
                dest = source.with_name('deduplicated-' + source.name)
                migrate(source, dest)
                self.assertEqual(original, [json.loads(line) for line in dest.read_text().splitlines()])
                for row in original:
                    for field in ('source_qa_id', 'id', 'chunk_index'):
                        self.assertNotIn(field, row)
                    self.assertEqual(row['source_metadata'], {'qa_id': ROW['qa_id'], 'id': None, 'chunk_index': 2})
                cfg['output'][family]['path'] = str(dest)
            resumed = pipeline(cfg)
            try:
                await resumed.run([ROW])
                self.assertEqual(resumed.generator.calls, [])
                for family in ('qwen3_8', 'llm_jp_4'):
                    self.assertEqual(resumed.stats[family + '_skipped'], 3)
                    self.assertEqual(len(resumed.outputs[family].records), 3)
            finally:
                await resumed.aclose()
            # Missing metadata must stop before generation or duplicate writes.
            path = Path(cfg['output']['qwen3_8']['path'])
            before = path.read_bytes()
            Path(str(path) + '.resume.jsonl').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing output resume metadata'):
                pipeline(cfg)
            self.assertEqual(path.read_bytes(), before)

    async def test_migration_failure_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory) / 'old.jsonl', Path(directory) / 'new.jsonl'
            source.write_text('{broken\n')
            before = source.read_bytes()
            with self.assertRaises(ValueError):
                migrate(source, dest)
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse(dest.exists())
            self.assertFalse(Path(str(dest) + '.resume.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
