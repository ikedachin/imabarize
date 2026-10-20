import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import httpx
import yaml

from reasoning_effort.output_schema import compact_output, template_kwargs
from temporary_tools.standardize_reasoning_dataset.convert import ConversionJournal, Translator, run, validate_translation
from test.test_reasoning_effort_dataset import FakeTokenizer, THINKING

ROOT = Path(__file__).resolve().parents[1]
TRANSLATED = {'thinking': THINKING.replace('問われとる内容を確認するんよ。', '問われている内容を確認します。')
              .replace('条件が合うけん、この答えになるね。', '条件が合うため、この答えになります。'),
              'answer': '藤堂高虎です。'}


def row(family):
    field = 'reasoning_content' if family == 'qwen3_8' else 'thinking'
    effort = 'xhigh' if family == 'qwen3_8' else 'high'
    value = dict(qa_id=f'original:{family}', question='築城者は誰？', answer='藤堂高虎なんよ。',
                 thinking=THINKING, reasoning_effort=effort, eval=5, thinking_tokens=999,
                 thinking_generator='original-model', source_metadata={'qa_id': 'source', 'id': 1, 'chunk_index': 0},
                 chat_template_kwargs=template_kwargs(family, effort))
    value['messages'] = [{'role': 'user', 'content': value['question']},
                         {'role': 'assistant', 'content': value['answer'], field: value['thinking']}]
    return value


class FakeTranslator:
    def __init__(self):
        self.calls = 0

    async def translate(self, source):
        self.calls += 1
        return deepcopy(TRANSLATED)


class ConversionTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrency_limit_duplicate_rows_failure_and_resume(self):
        class SlowTranslator:
            def __init__(self):
                self.active = self.peak = self.calls = 0

            async def translate(self, source):
                self.calls += 1
                self.active += 1
                self.peak = max(self.peak, self.active)
                try:
                    await asyncio.sleep(0.01)
                    if source['question'] == 'failure':
                        raise ValueError('test failure')
                    return deepcopy(TRANSLATED)
                finally:
                    self.active -= 1

        for limit in (1, 3):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as directory:
                cfg = self.config(directory)
                cfg['max_in_flight'] = limit
                for dataset in cfg['datasets']:
                    rows = []
                    for question in ['same', 'same', 'b', 'c', 'd', 'failure']:
                        value = row(dataset['family'])
                        value['question'] = question
                        value['messages'][0]['content'] = question
                        rows.append(value)
                    Path(dataset['input']).write_text(''.join(json.dumps(r) + '\n' for r in rows))
                translator = SlowTranslator()
                tokenizers = {f: FakeTokenizer() for f in ('qwen3_8', 'llm_jp_4')}
                result = await run(cfg, translator=translator, tokenizers=tokenizers)
                self.assertEqual(translator.peak, limit)
                self.assertEqual(translator.calls, 6)  # Four shared successes, two failures.
                self.assertEqual(result['written'], 8)
                self.assertEqual(result['failed'], 2)
                for dataset in cfg['datasets']:
                    records = [json.loads(line) for line in Path(dataset['output']).read_text().splitlines()]
                    self.assertEqual(len({r['qa_id'] for r in records}), 4)
                    for record in records:
                        self.assertEqual(compact_output(record), record)
                cfg['max_in_flight'] = 2
                resumed = await run(cfg, translator=translator, tokenizers=tokenizers)
                self.assertEqual(resumed.get('written', 0), 0)
                self.assertEqual(translator.calls, 8)  # Only the failures are attempted again.

    async def test_invalid_concurrency_rejected(self):
        for value in (0, -1, True, 1.5, '8', None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'max_in_flight'):
                    await run({'max_in_flight': value})

    def test_journal_uses_full_qa_id_and_preserves_existing_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'output.jsonl'
            journal = ConversionJournal(path)
            for family in ('qwen3_8', 'llm_jp_4'):
                journal.append(row(family))
            reopened = ConversionJournal(path)
            self.assertEqual(set(reopened.records), {'original:qwen3_8', 'original:llm_jp_4'})
            self.assertTrue(all('journal_key' not in r for r in reopened.records.values()))
            cache_path = Path(directory) / 'cache.jsonl'
            cache_path.write_text(json.dumps({'journal_key': 'old-key', 'translation': TRANSLATED}))
            cache = ConversionJournal(cache_path)
            self.assertEqual(cache.records['old-key']['translation'], TRANSLATED)
            cache.append({'translation': TRANSLATED}, 'new-key')
            self.assertEqual(set(ConversionJournal(cache_path).records), {'old-key', 'new-key'})
            # Only a damaged final line may be repaired; earlier corruption is fatal.
            path.write_bytes(b'broken\n{}\n')
            with self.assertRaises(ValueError):
                ConversionJournal(path)

    def config(self, directory):
        cfg = yaml.safe_load((ROOT / 'temporary_tools/standardize_reasoning_dataset/settings.yaml').read_text())
        cfg['prompt'] = str(ROOT / 'temporary_tools/standardize_reasoning_dataset/prompt.md')
        cfg['cache_path'] = str(Path(directory) / 'cache.jsonl')
        cfg['failures_path'] = str(Path(directory) / 'failures.jsonl')
        for d in cfg['datasets']:
            d['input'] = str(Path(directory) / (d['family'] + '-input.jsonl'))
            d['output'] = str(Path(directory) / (d['family'] + '-output.jsonl'))
            Path(d['input']).write_text(json.dumps(row(d['family'])) + '\n')
        return cfg

    async def test_shared_conversion_resume_and_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory)
            translator = FakeTranslator()
            tokenizers = {f: FakeTokenizer() for f in ('qwen3_8', 'llm_jp_4')}
            originals = [Path(d['input']).read_bytes() for d in cfg['datasets']]
            stats = await run(cfg, translator=translator, tokenizers=tokenizers)
            self.assertEqual(stats['written'], 2)
            self.assertEqual(translator.calls, 1)
            for d in cfg['datasets']:
                result = json.loads(Path(d['output']).read_text())
                self.assertEqual(compact_output(result), result)
                self.assertEqual(result['answer'], TRANSLATED['answer'])
                self.assertEqual(result['thinking'], TRANSLATED['thinking'])
                self.assertNotEqual(result['qa_id'], row(d['family'])['qa_id'])
                self.assertEqual(result['reasoning_effort'], row(d['family'])['reasoning_effort'])
                self.assertIsNone(result['eval'])
                self.assertEqual(result['thinking_tokens'], len(TRANSLATED['thinking'].split()))
            before = [Path(d['output']).read_bytes() for d in cfg['datasets']]
            stats = await run(cfg, translator=translator, tokenizers=tokenizers)
            self.assertEqual(stats['skipped'], 2)
            self.assertEqual(translator.calls, 1)
            self.assertEqual(before, [Path(d['output']).read_bytes() for d in cfg['datasets']])
            self.assertEqual(originals, [Path(d['input']).read_bytes() for d in cfg['datasets']])
            # A partial final write is repaired; the cached conversion recovers the row.
            Path(cfg['datasets'][0]['output']).write_bytes(b'{"qa_id":')
            stats = await run(cfg, translator=translator, tokenizers=tokenizers)
            self.assertEqual(stats['written'], 1)
            self.assertEqual(translator.calls, 1)

    async def test_bad_input_continues_and_path_collision_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory)
            path = Path(cfg['datasets'][0]['input'])
            path.write_text('broken\n' + path.read_text())
            stats = await run(cfg, translator=FakeTranslator(), tokenizers={f: FakeTokenizer() for f in ('qwen3_8', 'llm_jp_4')})
            self.assertEqual(stats['failed'], 1)
            self.assertEqual(stats['written'], 2)
            cfg['datasets'][0]['output'] = cfg['datasets'][1]['input']
            with self.assertRaises(ValueError):
                await run(cfg)

    def test_structural_and_number_changes_rejected(self):
        source = row('qwen3_8')
        validate_translation(source, TRANSLATED)
        for result in (dict(TRANSLATED, answer='1600年です。'),
                       dict(TRANSLATED, thinking=TRANSLATED['thinking'].replace('質問の整理', '質問')),
                       dict(TRANSLATED, answer=''), {'thinking': 'missing answer'}):
            with self.assertRaises(ValueError):
                validate_translation(source, result)

    async def test_api_invalid_json_retried_without_imabari_system(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory)
            cfg['wait_seconds'] = 0
            translator = Translator(cfg, Path(cfg['prompt']).read_text())
            await translator.client.aclose()
            calls = []
            def respond(request):
                calls.append(json.loads(request.content))
                content = 'invalid' if len(calls) == 1 else json.dumps(TRANSLATED)
                return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]})
            translator.client = httpx.AsyncClient(base_url='http://test/', transport=httpx.MockTransport(respond))
            try:
                self.assertEqual(await translator.translate(row('qwen3_8')), TRANSLATED)
                self.assertEqual(len(calls), 2)
                self.assertIn('標準語', calls[0]['messages'][0]['content'])
                self.assertIn('検証エラー', calls[1]['messages'][1]['content'])
            finally:
                await translator.aclose()
