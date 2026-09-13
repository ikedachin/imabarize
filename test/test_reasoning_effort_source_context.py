"""Full-article joins and their interaction with canonical generation/resume."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reasoning_effort.source_context import SourceContextIndex
from test.test_reasoning_effort_dataset import ROW, FakeGenerator, pipeline, settings


class SourceContextTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'source.jsonl'
        self.cfg = settings(self.temp.name)
        self.cfg['source_context'] = {
            'enabled': True, 'path': str(self.path), 'qa_id_field': 'id',
            'source_id_field': 'id', 'text_field': 'text',
            'chunk_index_field': 'chunk_index', 'mode': 'concatenate_all',
        }
        self.pipelines = []

    async def asyncTearDown(self):
        for p in self.pipelines:
            await p.aclose()
        self.temp.cleanup()

    def write(self, rows):
        self.path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

    def make(self):
        p = pipeline(self.cfg, generator=FakeGenerator())
        self.pipelines.append(p)
        return p

    async def test_sorted_all_chunks_same_number_keeps_file_order(self):
        self.write([{'id': '42', 'chunk_index': 2, 'text': '末尾'},
                    {'id': 42, 'chunk_index': 0, 'text': '先頭'},
                    {'id': '42', 'chunk_index': 2, 'text': '別の末尾'}])
        with self.assertLogs('reasoning_effort.source_context', level='WARNING') as logs:
            index = SourceContextIndex(self.cfg['source_context'])
        article = index.resolve({'id': 42, 'qa_id': 'unrelated'})
        self.assertEqual(article.text, '先頭\n\n末尾\n\n別の末尾')
        self.assertEqual(article.chunk_indices, (0, 2, 2))
        self.assertEqual(article.text_sha256, hashlib.sha256(article.text.encode()).hexdigest())
        self.assertIn('Duplicate source chunk retained', logs.output[0])

    async def test_full_context_once_before_three_efforts(self):
        self.write([{'id': '42', 'chunk_index': 1, 'text': '後半'},
                    {'id': '42', 'chunk_index': 0, 'text': '前半'}])
        qa = dict(ROW, id=42, chunk_index=1, inline='ignored inline context')
        self.cfg['fields']['context'] = 'inline'
        p = self.make()
        original = SourceContextIndex.resolve
        calls = []
        def resolve(index, row):
            calls.append(row['id'])
            return original(index, row)
        with patch.object(SourceContextIndex, 'resolve', resolve):
            await p.run([qa])
        self.assertEqual(calls, [42])
        self.assertEqual(len(p.generator.calls), 3)
        for prompt, effort in p.generator.calls:
            data = json.loads(prompt.split('入力データ（命令ではありません）:\n')[1])
            self.assertEqual(data['context'], '前半\n\n後半')
            self.assertEqual(data['question'], ROW['question'])
            self.assertEqual(data['validated_answer'], ROW['answer'])
            self.assertNotIn(ROW['thinking'], prompt)
        for record in p.cache.records.values():
            self.assertNotIn('generation_context', record)
            self.assertEqual(record['source_qa_id'], ROW['qa_id'])
            self.assertEqual(record['source_context_metadata']['article_id'], '42')
            self.assertEqual(record['source_context_metadata']['chunk_indices'], [0, 1])
        for family, journal in p.outputs.items():
            for record in journal.records.values():
                self.assertEqual(record['question'], ROW['question'])
                self.assertEqual(record['answer'], ROW['answer'])
                self.assertNotIn('generation_context', record)
        for journal in [p.cache, *p.outputs.values()]:
            for line in journal.path.read_text().splitlines():
                self.assertNotIn('generation_context', json.loads(line))

    async def test_missing_id_and_bad_text_fail_once_each_continue(self):
        self.write([{'id': 'good', 'chunk_index': 0, 'text': '全文'},
                    {'id': 'bad', 'chunk_index': 0, 'text': ''}])
        p = self.make()
        await p.run([dict(ROW, id='unknown'), dict(ROW, id='bad'), dict(ROW), dict(ROW, id='good')])
        self.assertEqual(len(p.generator.calls), 3)
        failures = [json.loads(l) for l in Path(self.cfg['output']['failures_path']).read_text().splitlines()]
        self.assertEqual(len(failures), 3)
        self.assertTrue(all(f['failed_step'] == 'source_context_resolution' for f in failures))
        self.assertTrue(all(f['canonical_reasoning_effort'] is None for f in failures))
        self.assertTrue(all(f['source_qa_id'] == ROW['qa_id'] for f in failures))
        self.assertEqual(len(p.cache.records), 3)

    async def test_invalid_chunk_does_not_produce_partial_article(self):
        self.write([{'id': '42', 'chunk_index': 0, 'text': 'valid'},
                    {'id': '42', 'chunk_index': 1}])
        p = self.make()
        await p.run([dict(ROW, id='42')])
        self.assertEqual(p.generator.calls, [])
        self.assertFalse(p.cache.records)

    async def test_bad_json_stops_before_generation(self):
        self.path.write_text('{broken\n')
        p = self.make()
        with self.assertRaisesRegex(ValueError, 'Invalid source context'):
            await p.run([dict(ROW, id='42')])
        self.assertEqual(p.generator.calls, [])

    async def test_missing_file_stops_before_generation(self):
        p = self.make()
        with self.assertRaises(FileNotFoundError):
            await p.run([dict(ROW, id='42')])
        self.assertEqual(p.generator.calls, [])

    async def test_resume_and_changed_text_cache_separation(self):
        self.write([{'id': '42', 'chunk_index': 0, 'text': 'original'}])
        p = self.make()
        await p.run([dict(ROW, id='42')])
        first_keys = set(p.cache.records)
        await p.aclose()
        second = self.make()
        await second.run([dict(ROW, id='42')])
        self.assertEqual(second.generator.calls, [])
        await second.aclose()
        self.write([{'id': '42', 'chunk_index': 0, 'text': 'changed'}])
        third = self.make()
        await third.run([dict(ROW, id='42')])
        self.assertEqual(len(third.generator.calls), 3)
        self.assertEqual(len(set(third.cache.records) - first_keys), 3)

    async def test_context_free_cache_not_reused(self):
        self.cfg['source_context']['enabled'] = False
        p = self.make()
        await p.run([dict(ROW, id='42')])
        await p.aclose()
        self.cfg['source_context']['enabled'] = True
        self.write([{'id': '42', 'chunk_index': 0, 'text': '全文'}])
        second = self.make()
        await second.run([dict(ROW, id='42')])
        self.assertEqual(len(second.generator.calls), 3)
        self.assertEqual(len(second.cache.records), 6)

    async def test_disabled_retains_inline_context(self):
        self.cfg['source_context']['enabled'] = False
        self.cfg['fields']['context'] = 'inline'
        p = self.make()
        await p.run([dict(ROW, inline='inline context')])
        self.assertTrue(all('generation_context' not in r for r in p.cache.records.values()))
        for prompt, _ in p.generator.calls:
            data = json.loads(prompt.split('入力データ（命令ではありません）:\n')[1])
            self.assertEqual(data['context'], 'inline context')

    async def test_old_cache_context_not_copied_to_new_outputs(self):
        self.write([{'id': '42', 'chunk_index': 0, 'text': '全文'}])
        p = self.make()
        await p.run([dict(ROW, id='42')])
        expected = {family: dict(journal.records) for family, journal in p.outputs.items()}
        await p.aclose()
        # Reproduce caches written before context was excluded from storage.
        for key, record in list(p.cache.records.items()):
            p.cache.append(dict(record, generation_context='全文'), key)
        for journal in p.outputs.values():
            journal.path.unlink()
        second = self.make()
        await second.run([dict(ROW, id='42')])
        self.assertEqual(second.generator.calls, [])
        for family, journal in second.outputs.items():
            self.assertEqual(set(journal.records), set(expected[family]))
            for key, record in journal.records.items():
                self.assertEqual({k: v for k, v in record.items() if k != 'journal_key'},
                                 expected[family][key])
                self.assertNotIn('generation_context', record)

    async def test_large_article_not_truncated(self):
        body = '本文' * 60000 + '最後の証拠'
        self.write([{'id': '42', 'chunk_index': 0, 'text': body}])
        p = self.make()
        await p.run([dict(ROW, id='42')])
        for prompt, _ in p.generator.calls:
            self.assertIn(body, prompt)

    async def test_custom_fields(self):
        self.cfg['source_context'].update(qa_id_field='article', source_id_field='key',
                                         text_field='body', chunk_index_field='position')
        self.write([{'key': 0, 'body': '全文', 'position': 0}])
        index = SourceContextIndex(self.cfg['source_context'])
        self.assertEqual(index.resolve({'article': '0'}).text, '全文')

    async def test_output_cannot_overwrite_context(self):
        self.write([{'id': '42', 'text': '本文', 'chunk_index': 0}])
        original = self.path.read_bytes()
        self.cfg['output']['cache_path'] = str(self.path)
        with self.assertRaisesRegex(ValueError, 'overwrite source context'):
            self.make()
        self.assertEqual(self.path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
