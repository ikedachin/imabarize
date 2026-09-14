import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest

import httpx
import yaml

from main_create_reasoning_effort_dataset import load_source
from pipelines.create_reasoning_effort_dataset import ReasoningEffortDatasetPipeline
from reasoning_effort.cache import JsonlJournal
from reasoning_effort.formatters import Qwen38ReasoningEffortFormatter, LLMJP4ReasoningEffortFormatter
from reasoning_effort.generator import ThinkingGenerator
from reasoning_effort.validators import ThinkingFormatValidator, classify_length, count_tokens, length_errors, retry_feedback

ROOT = Path(__file__).resolve().parents[1]
THINKING = '## 思考プロセス\n\n### 1. 質問の整理\n\n問われとる内容を確認するんよ。\n\n### 2. 結論への整理\n\n条件が合うけん、この答えになるね。'
ROW = {'qa_id': 'source-1', 'question': '今治城を築いたのは誰？', 'answer': '藤堂高虎なんよ。',
       'thinking': 'OLD THINKING MUST NOT ENTER PROMPT', 'eval': '5', 'source_files': ['a'], 'chunk_index': 2}


class FakeTokenizer:
    def __init__(self, multiplier=1):
        self.multiplier = multiplier
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [0] * (len(text.split()) * self.multiplier)

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        assert kwargs['add_generation_prompt'] is False
        a = messages[-1]
        e = kwargs['reasoning_effort']
        if 'reasoning_content' in a:
            condition = '' if e == 'medium' else f'Reasoning effort is set to {e}.\n'
            return condition + f"<think>\n{a['reasoning_content']}\n</think>\n\n{a['content'].strip()}<|im_end|>"
        return (f"Reasoning: {e}\n<|start|>assistant<|channel|>analysis<|message|>{a['thinking']}<|end|>"
                f"<|start|>assistant<|channel|>final<|message|>{a['content']}<|return|>")


class FakeGenerator:
    def __init__(self, outputs=None):
        self.calls = []
        self.outputs = list(outputs or [])
        self.active = self.peak = 0

    async def generate(self, prompt, effort):
        self.calls.append((prompt, effort))
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.001)
        self.active -= 1
        return self.outputs.pop(0) if self.outputs else THINKING


def settings(directory):
    cfg = yaml.safe_load((ROOT / 'yamls/create_reasoning_effort_dataset_settings_format.yaml').read_text())
    cfg['source_context'] = {'enabled': False}
    for effort in ('low', 'medium', 'high'):
        cfg['thinking_generation'][effort].update(min_tokens=1, max_tokens=1000,
            prompt=str(ROOT / f'prompts/create_reasoning_effort_dataset/thinking_{effort}.md'))
    for family in ('qwen3_8', 'llm_jp_4'):
        cfg['output'][family]['path'] = str(Path(directory) / f'{family}.jsonl')
    cfg['output']['cache_path'] = str(Path(directory) / 'cache.jsonl')
    cfg['output']['failures_path'] = str(Path(directory) / 'failures.jsonl')
    cfg['wait_seconds'] = 0
    cfg['max_in_flight'] = 2
    return cfg


def pipeline(cfg, generator=None, formatters=None):
    return ReasoningEffortDatasetPipeline(cfg, generator=generator or FakeGenerator(),
        reference_tokenizer=FakeTokenizer(3), formatters=formatters or {
            'qwen3_8': Qwen38ReasoningEffortFormatter(FakeTokenizer(1), 'qwen'),
            'llm_jp_4': LLMJP4ReasoningEffortFormatter(FakeTokenizer(2), 'llmjp'),
        })


class ValidatorTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(ThinkingFormatValidator().validate(THINKING), [])

    def test_leading_response_separator_remains_invalid(self):
        self.assertIn('first_line_mismatch', ThinkingFormatValidator().validate('\n\n' + THINKING))
        self.assertIn('最初の文字を #', retry_feedback(['first_line_mismatch']))

    def test_invalid_formats(self):
        cases = [
            ('', 'empty_thinking'),
            ('\n' + THINKING, 'first_line_mismatch'),
            (THINKING.replace('## 思考プロセス\n\n', '## 思考プロセス\n'), 'missing_blank_line_after_title'),
            (THINKING.replace('## 思考プロセス\n\n', '## 思考プロセス\n\n\n'), 'multiple_blank_lines_after_title'),
            (THINKING.replace('質問の整理\n\n', '質問の整理\n'), 'missing_blank_line_after_section_heading'),
            (THINKING.replace('質問の整理\n\n', '質問の整理\n\n\n'), 'multiple_blank_lines_after_section_heading'),
            (THINKING.replace('んよ。\n\n###', 'んよ。\n###'), 'missing_blank_line_before_next_section'),
            (THINKING.replace('んよ。\n\n###', 'んよ。\n\n\n###'), 'multiple_blank_lines_before_next_section'),
            (THINKING.replace('### 2.', '### 3.'), 'section_number_not_contiguous'),
            (THINKING.replace('### 1.', '### 0.'), 'invalid_first_section_heading'),
            (THINKING.replace('### 1.', '###1.'), 'invalid_section_heading'),
            (THINKING.replace('### 1.', '### 1'), 'invalid_section_heading'),
            (THINKING.replace('### 1.', '###'), 'invalid_section_heading'),
            (THINKING.replace('問われとる内容を確認するんよ。\n\n', ''), 'empty_section_body'),
            (THINKING.split('\n\n### 2.')[0], 'too_few_sections'),
            ('<think>' + THINKING + '</think>', 'think_tag_found'),
            (THINKING + '\n```', 'code_fence_found'),
            (THINKING + '\n\n最終回答:\n答え', 'final_answer_found'),
            (THINKING + '\n<|return|>', 'final_answer_or_control_tag_found'),
        ]
        for text, expected in cases:
            with self.subTest(expected=expected, text=text):
                self.assertIn(expected, ThinkingFormatValidator().validate(text))

    def test_length(self):
        self.assertEqual(length_errors(256, {'min_tokens': 64, 'max_tokens': 256}), [])
        self.assertEqual(length_errors(63, {'min_tokens': 64}), ['thinking_too_short'])
        self.assertEqual(length_errors(257, {'max_tokens': 256}), ['thinking_too_long'])
        ranges = {'low': {'max_tokens': 256}, 'medium': {'min_tokens': 257, 'max_tokens': 768}, 'high': {'min_tokens': 769}}
        for count, effort in [(0, 'low'), (256, 'low'), (257, 'medium'), (768, 'medium'), (769, 'high')]:
            self.assertEqual(classify_length(count, ranges), effort)

    def test_prompts(self):
        for effort in ('low', 'medium', 'high'):
            prompt = (ROOT / f'prompts/create_reasoning_effort_dataset/thinking_{effort}.md').read_text()
            for phrase in ('今治弁', '空行を1行', '最低2section', '<think>禁止', '架空の根拠'):
                self.assertIn(phrase, prompt)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cfg = settings(self.temp.name)
        self.pipelines = []

    async def asyncTearDown(self):
        for p in self.pipelines:
            await p.aclose()
        self.temp.cleanup()

    def make(self, **kwargs):
        p = pipeline(self.cfg, **kwargs)
        self.pipelines.append(p)
        return p

    async def test_dual_expand_and_mapping(self):
        g = FakeGenerator()
        p = self.make(generator=g)
        summary = await p.run([ROW])
        self.assertEqual(len(g.calls), 3)
        self.assertEqual(summary['output_records'], {'qwen3_8': 3, 'llm_jp_4': 3})
        self.assertEqual(g.peak, 2)
        for key, q in p.outputs['qwen3_8'].records.items():
            j = p.outputs['llm_jp_4'].records[key]
            for field in ('source_qa_id', 'question', 'thinking', 'answer'):
                self.assertEqual(q[field], j[field])
            self.assertEqual(q['source_metadata'], {k: ROW.get(k) for k in ('qa_id', 'id', 'chunk_index')})
            self.assertEqual(q['qa_id'], key + ':qwen3_8')
            self.assertEqual(j['reasoning_effort'], p.cache.records[key]['canonical_reasoning_effort'])
            self.assertEqual(q['reasoning_effort'], {'low': 'low', 'medium': 'medium', 'high': 'xhigh'}[j['reasoning_effort']])
            self.assertEqual(j['thinking_tokens'], q['thinking_tokens'] * 2)
            self.assertEqual(p.cache.records[key]['canonical_thinking_tokens'], q['thinking_tokens'] * 3)
        self.assertTrue(all(ROW['thinking'] not in prompt for prompt, _ in g.calls))

    async def test_fixed(self):
        self.cfg['reasoning_effort'].update(mode='fixed', effort='high')
        g = FakeGenerator()
        await self.make(generator=g).run([ROW])
        self.assertEqual(len(g.calls), 1)
        self.assertEqual(g.calls[0][1], 'high')

    async def test_token_length(self):
        self.cfg['reasoning_effort']['mode'] = 'token_length'
        p = self.make()
        await p.run([ROW])
        record = next(iter(p.cache.records.values()))
        self.assertEqual(record['canonical_reasoning_effort'], 'low')
        self.assertEqual(record['generation_method'], 'token_length_labeled')

    async def test_resume_failed_formatter_only(self):
        class FailHigh(LLMJP4ReasoningEffortFormatter):
            def format(self, record):
                if record['canonical_reasoning_effort'] == 'high':
                    raise ValueError('simulated target failure')
                return super().format(record)
        p = self.make(formatters={'qwen3_8': Qwen38ReasoningEffortFormatter(FakeTokenizer(), 'qwen'),
                                  'llm_jp_4': FailHigh(FakeTokenizer(2), 'llmjp')})
        await p.run([ROW])
        self.assertEqual(len(p.cache.records), 3)
        self.assertEqual(len(p.outputs['llm_jp_4'].records), 2)
        await p.aclose()
        g = FakeGenerator()
        second = self.make(generator=g)
        result = await second.run([ROW])
        self.assertEqual(len(g.calls), 0)
        self.assertEqual(result['llm_jp_4_written'], 1)
        self.assertEqual(result['qwen3_8_skipped'], 3)
        self.assertEqual(len(second.outputs['qwen3_8'].records), 3)

    async def test_format_retry_feedback(self):
        self.cfg['reasoning_effort'].update(mode='fixed', effort='low')
        g = FakeGenerator(['broken', THINKING])
        p = self.make(generator=g)
        await p.run([ROW])
        self.assertEqual(len(g.calls), 2)
        self.assertIn('first_line_mismatch', g.calls[1][0])
        self.assertEqual(len(p.cache.records), 1)

    async def test_length_retry(self):
        self.cfg['reasoning_effort'].update(mode='fixed', effort='low')
        normal = count_tokens(FakeTokenizer(3), THINKING)
        self.cfg['thinking_generation']['low']['max_tokens'] = normal
        g = FakeGenerator([THINKING + ' extra words', THINKING])
        p = self.make(generator=g)
        await p.run([ROW])
        self.assertEqual(len(g.calls), 2)
        self.assertIn('thinking_too_long', g.calls[1][0])
        self.assertIn('前回の実測はreference tokenizerで', g.calls[1][0])

    async def test_invalid_never_cached_or_written(self):
        self.cfg['reasoning_effort'].update(mode='fixed', effort='low')
        p = self.make(generator=FakeGenerator(['bad'] * 3))
        result = await p.run([ROW])
        self.assertEqual(result['generator_calls'], 3)
        self.assertFalse(p.cache.records)
        self.assertFalse(p.outputs['qwen3_8'].records)
        failure = json.loads(Path(self.cfg['output']['failures_path']).read_text())
        self.assertEqual(failure['failed_step'], 'thinking_format_validation')
        self.assertEqual(failure['rejected_thinking'], 'bad')

    async def test_missing_record_does_not_stop_batch(self):
        p = self.make()
        await p.run([{}, ROW])
        self.assertEqual(len(p.cache.records), 3)

    async def test_duplicate_source_no_duplicate_calls(self):
        g = FakeGenerator()
        p = self.make(generator=g)
        await p.run([ROW, ROW])
        self.assertEqual(len(g.calls), 3)
        self.assertEqual(len(p.outputs['qwen3_8'].records), 3)

    async def test_input_mapping(self):
        self.cfg['fields'] = {'id': 'uid', 'question': 'q', 'answer': 'a', 'thinking': 't', 'context': 'ctx'}
        self.cfg['output']['keep_original_thinking'] = False
        p = self.make()
        await p.run([{'uid': 0, 'q': 'Q', 'a': 'A', 't': 'old', 'ctx': 'context'}])
        record = next(iter(p.cache.records.values()))
        self.assertEqual(record['source_qa_id'], 0)
        self.assertNotIn('t', record)
        self.assertNotIn('original_thinking', record)
        self.assertIn('context', p.generator.calls[0][0])

    async def test_corrupt_cached_thinking_fails_without_inference(self):
        p = self.make()
        await p.run([ROW])
        await p.aclose()
        cache_path = Path(self.cfg['output']['cache_path'])
        records = [json.loads(x) for x in cache_path.read_text().splitlines()]
        records[0]['thinking'] = 'broken'
        cache_path.write_text(''.join(json.dumps(x) + '\n' for x in records))
        g = FakeGenerator()
        second = self.make(generator=g)
        await second.run([ROW])
        self.assertEqual(g.calls, [])
        self.assertEqual(second.stats['thinking_format_validation_terminal_failures'], 1)

    async def test_length_exhaustion_never_cached(self):
        self.cfg['reasoning_effort'].update(mode='fixed', effort='low')
        self.cfg['thinking_generation']['low']['max_tokens'] = 1
        p = self.make()
        await p.run([ROW])
        self.assertEqual(len(p.generator.calls), 3)
        self.assertFalse(p.cache.records)
        self.assertFalse(p.outputs['llm_jp_4'].records)

    async def test_write_failure_resume_without_inference(self):
        p = self.make()
        def fail_write(record):
            raise OSError('simulated disk failure')
        p.outputs['llm_jp_4'].append = fail_write
        await p.run([ROW])
        self.assertEqual(len(p.cache.records), 3)
        self.assertEqual(p.stats['llm_jp_4_write_terminal_failures'], 3)
        await p.aclose()
        g = FakeGenerator()
        second = self.make(generator=g)
        await second.run([ROW])
        self.assertEqual(g.calls, [])
        self.assertEqual(second.stats['llm_jp_4_written'], 3)

    async def test_new_target_reuses_canonical(self):
        self.cfg['targets']['llm_jp_4']['enabled'] = False
        p = self.make()
        await p.run([ROW])
        await p.aclose()
        self.cfg['targets']['llm_jp_4']['enabled'] = True
        g = FakeGenerator()
        second = self.make(generator=g)
        await second.run([ROW])
        self.assertEqual(g.calls, [])
        self.assertEqual(second.stats['llm_jp_4_written'], 3)

    async def test_generation_exception_isolated(self):
        class FailLow(FakeGenerator):
            async def generate(self, prompt, effort):
                if effort == 'low':
                    raise ValueError('simulated inference failure')
                return await super().generate(prompt, effort)
        p = self.make(generator=FailLow())
        await p.run([ROW])
        self.assertEqual(len(p.cache.records), 2)
        self.assertEqual(p.stats['generation_terminal_failures'], 1)

    async def test_path_collision_rejected(self):
        self.cfg['output']['llm_jp_4']['path'] = self.cfg['output']['qwen3_8']['path']
        with self.assertRaises(ValueError):
            self.make()

    async def test_invalid_canonical_setting_rejected(self):
        self.cfg['reasoning_effort'].update(mode='fixed', effort='xhigh')
        with self.assertRaises(ValueError):
            self.make()


class StorageTests(unittest.TestCase):
    def test_interrupted_tail(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'cache.jsonl'
            j = JsonlJournal(path)
            j.append({'canonical_record_id': 'a'})
            with path.open('ab') as f:
                f.write(b'{"broken":')
            recovered = JsonlJournal(path)
            recovered.append({'canonical_record_id': 'b'})
            self.assertEqual(set(JsonlJournal(path).records), {'a', 'b'})

    def test_local_malformed_line_continues(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'in.jsonl'
            path.write_text('bad\n' + json.dumps(ROW) + '\n')
            rows = list(load_source({'source': {'type': 'local', 'path': str(path)}}))
            self.assertEqual(rows[1], ROW)


class HttpTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_and_effort_payload(self):
        payloads = []
        def handler(request):
            self.assertEqual(request.url.path, '/v1/chat/completions')
            payloads.append(json.loads(request.content))
            if len(payloads) == 1:
                return httpx.Response(429)
            return httpx.Response(200, json={'choices': [{'message': {'content': THINKING}}]})
        cfg = {'generator': {'server_url': 'http://test/v1', 'model_name': 'generator',
                             'generation': {'chat_template_kwargs': {'enable_thinking': False}},
                             'use_reasoning_effort': True, 'reasoning_effort_by_canonical': {'high': 'xhigh'}},
               'max_retries': 1, 'wait_seconds': 0}
        client = httpx.AsyncClient(base_url='http://test/v1/', transport=httpx.MockTransport(handler))
        gen = ThinkingGenerator(cfg, client)
        try:
            self.assertEqual(await gen.generate('prompt', 'high'), THINKING)
            self.assertEqual(gen.call_count, 2)
            self.assertEqual(payloads[-1]['reasoning_effort'], 'xhigh')
            self.assertEqual(payloads[-1]['chat_template_kwargs'], {'enable_thinking': False})
            self.assertEqual(payloads[-1]['messages'][0]['role'], 'system')
            self.assertIn('先頭の空白・空行・改行', payloads[-1]['messages'][0]['content'])
            self.assertEqual(payloads[-1]['messages'][1], {'role': 'user', 'content': 'prompt'})
        finally:
            await gen.aclose()


@unittest.skipUnless(os.environ.get('REASONING_REAL_TOKENIZERS'), 'Set REASONING_REAL_TOKENIZERS to local HF cache')
class RealTokenizerTests(unittest.TestCase):
    def test_official_templates_all_efforts(self):
        from transformers import AutoTokenizer
        cache = os.environ['REASONING_REAL_TOKENIZERS']
        q = AutoTokenizer.from_pretrained('Qwen/Qwen3.8-27B', cache_dir=cache, local_files_only=True)
        j = AutoTokenizer.from_pretrained('llm-jp/llm-jp-4-8b-thinking', cache_dir=cache, local_files_only=True, trust_remote_code=True)
        for effort in ('low', 'medium', 'high'):
            canonical = dict(ROW, thinking=THINKING, canonical_reasoning_effort=effort, canonical_record_id='sample')
            qr = Qwen38ReasoningEffortFormatter(q, 'qwen').format(canonical)
            jr = LLMJP4ReasoningEffortFormatter(j, 'llmjp').format(canonical)
            self.assertEqual(qr['thinking'], jr['thinking'])
            self.assertEqual(qr['thinking_tokens'], len(q.encode(THINKING, add_special_tokens=False)))
            self.assertEqual(jr['thinking_tokens'], len(j.encode(THINKING, add_special_tokens=False)))
            jt = j.apply_chat_template(jr['messages'], tokenize=False, add_generation_prompt=False, **jr['chat_template_kwargs'])
            qt = q.apply_chat_template(qr['messages'], tokenize=False, add_generation_prompt=False, reasoning_effort=qr['reasoning_effort'])
            self.assertIn('<|channel|>analysis', jt)
            self.assertIn('<think>', qt)
            self.assertEqual(qt, q.apply_chat_template(qr['messages'], tokenize=False,
                add_generation_prompt=False, reasoning_effort=qr['reasoning_effort'],
                enable_thinking=True, preserve_thinking=True))
            for r in (qr, jr):
                self.assertNotIn('text', r)
                self.assertNotIn('template_messages', r)
            self.assertNotIn('chat_template_kwargs', qr)
            self.assertEqual(jr['messages'], [
                {'role': 'user', 'content': ROW['question']},
                {'role': 'assistant', 'content': ROW['answer'], 'thinking': THINKING},
            ])



if __name__ == '__main__':
    unittest.main()
