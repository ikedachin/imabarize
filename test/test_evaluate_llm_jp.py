import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import yaml

from main_evaluate_llm_jp import main
from pipelines.evaluate_llm_jp import ChatClient, EvaluationError, evaluate, extract_final_answer, final_answer, load_dataset
from tool_recover_llm_jp_answers import recover


class ResponseParsingTests(unittest.TestCase):
    def test_cli_uses_stage_specific_completion_for_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Path(directory) / 'settings.yaml'
            settings.write_text(yaml.safe_dump({'eval_qa_path': 'input.jsonl'}), encoding='utf-8')

            async def fake_evaluate(config, dataset, stage):
                return {'remaining_generation_records': 0, 'remaining_records': 2, 'stage': stage}

            with patch('main_evaluate_llm_jp.evaluate', side_effect=fake_evaluate):
                with patch('sys.argv', ['main_evaluate_llm_jp.py', '-p', str(settings), '--stage', 'generate']):
                    self.assertEqual(main(), 0)
                with patch('sys.argv', ['main_evaluate_llm_jp.py', '-p', str(settings), '--stage', 'judge']):
                    self.assertEqual(main(), 1)

    def test_final_channel_and_think_are_not_sent_as_answer(self):
        raw = '<|channel|>analysis<|message|>秘密<|end|><|start|>assistant<|channel|>final<|message|>回答<|return|>'
        self.assertEqual(final_answer(raw, 'stop'), '回答')
        raw_imabari = '<|channel|>analysis_imabari<|message|>考える<|end|><|start|>assistant<|channel|>final_imabari<|message|>ほうなんよ<|return|>'
        self.assertEqual(extract_final_answer(raw_imabari, 'stop'), ('ほうなんよ', 'final_imabari'))
        self.assertEqual(extract_final_answer('analysis_imabari 思考 assistant final_imabari ほうなんよ。', 'stop'), ('ほうなんよ。', 'final_imabari'))
        self.assertEqual(extract_final_answer('assistant analysis 思考 assistant final 答え', 'stop'), ('答え', 'final'))
        self.assertEqual(extract_final_answer('assistant final 答え', 'stop'), ('答え', 'final'))
        self.assertEqual(extract_final_answer('質問に assistant final と書かれています。', 'stop'), ('質問に assistant final と書かれています。', None))
        self.assertEqual(final_answer('<think>考える</think>回答', 'stop'), '回答')
        for raw, reason in [('答え', 'length'), ('<think>思考だけ', 'stop'), ('<|channel|>analysis<|message|>思考', 'stop'), ('analysis_imabari 思考だけ', 'stop')]:
            with self.subTest(raw=raw), self.assertRaises(EvaluationError):
                final_answer(raw, reason)

    def test_recovery_copies_answers_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory) / 'source', Path(directory) / 'recovered'
            source.mkdir()
            (source / 'manifest.json').write_text('{"version": 3}', encoding='utf-8')
            original = [
                {'key': '1', 'answer': 'analysis_imabari 思考 assistant final_imabari 最終回答'},
                {'key': '2', 'answer': 'analysis 未完成'},
                {'key': '3', 'answer': '普通の回答', 'observed_final_channel': 'final'},
            ]
            original_text = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in original)
            (source / 'answers.jsonl').write_text(original_text, encoding='utf-8')
            self.assertEqual(recover(source, dest), (2, 1))
            self.assertEqual((source / 'answers.jsonl').read_text(encoding='utf-8'), original_text)
            saved = [json.loads(line) for line in (dest / 'answers.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual((saved[0]['answer'], saved[0]['observed_final_channel']), ('最終回答', 'final_imabari'))
            self.assertEqual(saved[1]['observed_final_channel'], 'final')
            self.assertEqual(json.loads((dest / 'rejected_answers.jsonl').read_text())['record']['key'], '2')
            with self.assertRaisesRegex(EvaluationError, 'new directory'):
                recover(source, dest)

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'eval.jsonl'
            path.write_text('\n'.join([json.dumps({'id': 'a', 'question': 'q', 'answer': 'a'})] * 2), encoding='utf-8')
            with self.assertRaisesRegex(EvaluationError, 'Duplicate'):
                load_dataset(path, {'id': 'id', 'question': 'question', 'answer': 'answer'})


class EndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def test_separate_stages_support_partial_resume_and_independent_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            dataset = base / 'eval.jsonl'
            dataset.write_text(json.dumps({'id': 'a', 'question': '質問', 'answer': '基準'}) + '\n', encoding='utf-8')
            prompt = base / 'judge.md'
            prompt.write_text('{question} {reference_answer} {candidate_answer}', encoding='utf-8')
            config = {'output_path': str(base / 'out'), 'judge_prompt': str(prompt), 'workers': 3,
                      'target': {'base_url': 'http://target/v1', 'model': 'target', 'api_key_env': 'TEST_TARGET_KEY', 'max_attempts': 1}}
            calls = {'target': 0, 'judge': 0}
            fail_once = True

            async def handler(request):
                nonlocal fail_once
                payload = json.loads(request.content)
                model = payload['model']
                calls[model] += 1
                if model == 'target' and fail_once:
                    fail_once = False
                    return httpx.Response(500, request=request)
                content = ('回答' if model == 'target' else '<judge_score>5</judge_score><judge_label>correct</judge_label><judge_reason>一致</judge_reason><style_judge_label>matches</style_judge_label><style_judge_reason>適切</style_judge_reason>')
                return httpx.Response(200, json={'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}, request=request)

            real_client = httpx.AsyncClient
            def create_client(*args, **kwargs):
                return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)
            with patch('pipelines.evaluate_llm_jp.httpx.AsyncClient', side_effect=create_client):
                with patch.dict(os.environ, {'TEST_TARGET_KEY': 'test'}):
                    generated = await evaluate(config, dataset, stage='generate')
                self.assertEqual((generated['generated_records'], generated['judged_records']), (5, 0))
                self.assertEqual(generated['remaining_generation_records'], 1)
                self.assertEqual(calls, {'target': 6, 'judge': 0})
                config['judge'] = {'base_url': 'http://judge/v1', 'model': 'judge'}
                with patch.dict(os.environ, {}, clear=True):
                    judged = await evaluate(config, dataset, stage='judge')
                self.assertEqual((judged['judged_records'], judged['remaining_records']), (5, 1))
                self.assertEqual(calls, {'target': 6, 'judge': 5})
                with patch.dict(os.environ, {'TEST_TARGET_KEY': 'test'}):
                    await evaluate(config, dataset, stage='generate')
                with patch.dict(os.environ, {}, clear=True):
                    final = await evaluate(config, dataset, stage='judge')
                self.assertEqual(final['remaining_records'], 0)
                self.assertEqual(calls, {'target': 7, 'judge': 6})
                changed = {**config, 'judge': {**config['judge'], 'model': 'other'}}
                with self.assertRaisesRegex(EvaluationError, 'Judge settings changed'):
                    await evaluate(changed, dataset, stage='judge')

    async def test_judge_requires_saved_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            dataset = base / 'eval.jsonl'
            dataset.write_text(json.dumps({'id': 'a', 'question': '質問', 'answer': '基準'}) + '\n', encoding='utf-8')
            config = {'output_path': str(base / 'out'), 'target': {'model': 'target'}, 'judge': {'model': 'judge'}}
            with self.assertRaisesRegex(EvaluationError, 'run --stage generate first'):
                await evaluate(config, dataset, stage='judge')

    async def test_six_conditions_reach_template_and_are_scored_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            dataset = base / 'eval.jsonl'
            dataset.write_text(json.dumps({'id': 'a', 'question': '質問', 'answer': '基準'}) + '\n', encoding='utf-8')
            prompt = base / 'judge.md'
            prompt.write_text('{question} {reference_answer} {candidate_answer}', encoding='utf-8')
            config = {'output_path': str(base / 'out'), 'judge_prompt': str(prompt), 'workers': 6,
                      'target': {'base_url': 'http://target/v1', 'model': 'target', 'max_in_flight': 3,
                                 'generation': {'chat_template_kwargs': {'conversation_start_date': '2026-09-11'}}},
                      'judge': {'base_url': 'http://judge/v1', 'model': 'judge', 'max_in_flight': 2}}
            payloads = []

            async def handler(request):
                payload = json.loads(request.content)
                if payload['model'] == 'target':
                    payloads.append(payload)
                    channels = payload['chat_template_kwargs']['valid_channels']
                    final = 'final_imabari' if 'final_imabari' in channels else 'final'
                    content = f'<|channel|>analysis<|message|>thinking<|end|><|start|>assistant<|channel|>{final}<|message|>答え<|return|>'
                else:
                    content = '<judge_score>5</judge_score><judge_label>correct</judge_label><judge_reason>一致</judge_reason><style_judge_label>matches</style_judge_label><style_judge_reason>指定どおり</style_judge_reason>'
                return httpx.Response(200, json={'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}, request=request)

            real_client = httpx.AsyncClient
            def create_client(*args, **kwargs):
                return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)
            with patch('pipelines.evaluate_llm_jp.httpx.AsyncClient', side_effect=create_client):
                stats = await evaluate(config, dataset)
                self.assertEqual(stats['input_questions'], 1)
                self.assertEqual(stats['total_records'], 6)
                self.assertEqual(stats['judged_records'], 6)
                combinations = {(p['chat_template_kwargs']['valid_channels'][0], p['chat_template_kwargs']['reasoning_effort']) for p in payloads}
                self.assertEqual(combinations, {(channel, effort) for channel in ('analysis', 'analysis_imabari') for effort in ('low', 'medium', 'high')})
                self.assertTrue(all(p['messages'] == [{'role': 'user', 'content': '質問'}] for p in payloads))
                self.assertTrue(all(p['chat_template_kwargs']['conversation_start_date'] == '2026-09-11' for p in payloads))
                self.assertTrue(all('generate_thinking' not in p['chat_template_kwargs'] for p in payloads))
                rows = [json.loads(line) for line in (base / 'out' / 'all.jsonl').read_text().splitlines()]
                self.assertEqual({row['observed_final_channel'] for row in rows}, {'final', 'final_imabari'})
                self.assertEqual(stats['by_style_and_effort']['imabari']['high']['style_labels']['matches'], 1)
                self.assertEqual(stats['by_style_and_effort']['imabari']['high']['observed_final_channels']['final_imabari'], 1)
                await evaluate(config, dataset)
                self.assertEqual(len(payloads), 6)

    async def test_retryable_status_is_retried(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, headers={'retry-after': '0'}, request=request)
            return httpx.Response(200, json={'choices': [{'message': {'content': '回答'}, 'finish_reason': 'stop'}]}, request=request)

        real_client = httpx.AsyncClient
        def create_client(*args, **kwargs):
            return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)
        with patch('pipelines.evaluate_llm_jp.httpx.AsyncClient', side_effect=create_client):
            client = ChatClient({'base_url': 'http://target/v1', 'model': 'target', 'max_attempts': 2})
            try:
                self.assertEqual(await client.complete('質問', extract_final=True), ('回答', None))
                self.assertEqual(calls, 2)
            finally:
                await client.close()

    async def test_worker_limits_resume_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            dataset = base / 'eval.jsonl'
            dataset.write_text(''.join(json.dumps({'id': str(i), 'question': f'q{i}', 'answer': 'ref'}) + '\n' for i in range(6)), encoding='utf-8')
            prompt = base / 'judge.md'
            prompt.write_text('{question} {reference_answer} {candidate_answer}', encoding='utf-8')
            config = {'output_path': str(base / 'out'), 'judge_prompt': str(prompt), 'workers': 6,
                      'styles': {'standard': {'label': '標準語', 'valid_channels': ['analysis', 'final', 'commentary']}},
                      'reasoning_efforts': ['medium'],
                      'target': {'base_url': 'http://target/v1', 'model': 'target', 'max_in_flight': 2, 'max_attempts': 1},
                      'judge': {'base_url': 'http://judge/v1', 'model': 'judge', 'max_in_flight': 1, 'max_attempts': 1}}
            running = {'target': 0, 'judge': 0}
            peak = {'target': 0, 'judge': 0}
            calls = {'target': 0, 'judge': 0}
            fail = True

            async def handler(request):
                nonlocal fail
                payload = json.loads(request.content)
                model = payload['model']
                calls[model] += 1
                running[model] += 1
                peak[model] = max(peak[model], running[model])
                await asyncio.sleep(0.005)
                running[model] -= 1
                if model == 'judge' and fail:
                    fail = False
                    return httpx.Response(401, request=request)
                content = ('<think>思考</think>回答' if model == 'target' else '<judge_score>5</judge_score><judge_label>correct</judge_label><judge_reason>一致</judge_reason><style_judge_label>unjudgeable</style_judge_label><style_judge_reason>短い</style_judge_reason>')
                return httpx.Response(200, json={'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}, request=request)

            real_client = httpx.AsyncClient
            def create_client(*args, **kwargs):
                return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)
            with patch('pipelines.evaluate_llm_jp.httpx.AsyncClient', side_effect=create_client):
                first = await evaluate(config, dataset)
                self.assertEqual(first['generated_records'], 6)
                self.assertEqual(first['judged_records'], 5)
                self.assertEqual(first['judge_failures'], 1)
                self.assertEqual(peak, {'target': 2, 'judge': 1})
                second = await evaluate(config, dataset)
                self.assertEqual(second['judged_records'], 6)
                self.assertEqual(second['generated_this_run'], 0)
                self.assertEqual(calls, {'target': 6, 'judge': 7})
                self.assertEqual(second['labels']['correct'], 6)
                self.assertEqual(second['mean_score'], 5)
                results = [json.loads(line) for line in (base / 'out' / 'all.jsonl').read_text().splitlines()]
                self.assertTrue(all(row['candidate_answer'] == '回答' for row in results))
                changed = {**config, 'target': {**config['target'], 'model': 'different'}}
                with self.assertRaisesRegex(EvaluationError, 'changed'):
                    await evaluate(changed, dataset)


if __name__ == '__main__':
    unittest.main()
