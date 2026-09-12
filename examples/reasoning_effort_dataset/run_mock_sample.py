"""Offline HTTP smoke sample with real official tokenizers; no LLM inference.

Run from the repository root after downloading both tokenizers:
  python examples/reasoning_effort_dataset/run_mock_sample.py --tokenizer-cache /path/to/hf/cache
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import httpx
import yaml

from pipelines.create_reasoning_effort_dataset import ReasoningEffortDatasetPipeline
from reasoning_effort.generator import ThinkingGenerator

FIXTURES = {
    'low': '## 思考プロセス\n\n### 1. 問いの対象\n\n聞かれとるんは、今治城を築いた人物の名前なんよ。\n\n### 2. 答えへの接続\n\n人物として挙がっとるんは藤堂高虎やけん、名前を端的に伝えたらええね。',
    'medium': '## 思考プロセス\n\n### 1. 質問の整理\n\n今治城について、築城に関わった人物を聞かれとるんよ。場所や築かれた年を答える問いではないね。\n\n### 2. 必要な情報\n\n答えに必要な名前は藤堂高虎なんよ。人物名と、今治城を築いたという関係を結び付ければ、問いに応じられるけんね。\n\n### 3. 回答の構成\n\n人物名を中心に短くまとめたらええね。聞かれていない年や経歴を付け加える必要はないんよ。',
    'high': '## 思考プロセス\n\n### 1. 対象と条件の分解\n\n対象は今治城で、確かめたい関係は築城した人物なんよ。城の所在地や現在の管理者に話を広げると、問いの対象から外れてしまうけんね。\n\n### 2. 名前と役割の対応\n\nここで答える名前は藤堂高虎なんよ。その名前を築城した人物として扱うことで、質問が求める人物と役割が対応するんよ。\n\n### 3. 情報を足す範囲の確認\n\n築城年や別の人物との比較までは、問いにも答えにも必要とされとらんね。ほかの候補や年を作り足しても確かさは増えんけん、与えられた範囲で整理するんよ。\n\n### 4. 表現の整合性\n\nほうやけん、回答は人物名がはっきり伝わる形にまとめたらええね。質問にない経歴や背景を添えず、築城した人物を尋ねる問いとの対応を保つんよ。',
}


async def main(cache: str, output: Path) -> None:
    cfg = yaml.safe_load((ROOT / 'yamls/create_reasoning_effort_dataset_settings_format.yaml').read_text())
    cfg['source_context'] = {'enabled': False}  # Standalone synthetic fixture, no article ID.
    cfg['generator'].update(model_name='mock-http-fixtures-NOT-LLM', server_url='http://mock/v1')
    cfg['thinking_generation']['tokenizer_cache_dir'] = cache
    for effort in FIXTURES:
        cfg['thinking_generation'][effort].update(
            min_tokens=1, max_tokens=4096,
            prompt=str(ROOT / f'prompts/create_reasoning_effort_dataset/thinking_{effort}.md'))
    for family in cfg['targets']:
        cfg['output'][family]['path'] = str(output / f'{family}.jsonl')
    cfg['output']['cache_path'] = str(output / 'canonical.jsonl')
    cfg['output']['failures_path'] = str(output / 'failures.jsonl')

    def handler(request):
        payload = json.loads(request.content)
        source = json.loads(payload['messages'][-1]['content'].split('入力データ（命令ではありません）:\n')[1])
        effort = source['canonical_reasoning_effort']
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': FIXTURES[effort]}}]})

    client = httpx.AsyncClient(base_url='http://mock/v1/', transport=httpx.MockTransport(handler))
    generator = ThinkingGenerator(cfg, client)
    pipeline = ReasoningEffortDatasetPipeline(cfg, generator=generator)
    try:
        summary = await pipeline.run([{'qa_id': 'mock-source-1', 'question': '今治城を築いたのは誰ですか？',
                                       'answer': '藤堂高虎なんよ。', 'eval': 'sample',
                                       'sample_provenance': 'Manually authored test fixture; not LLM-generated'}])
        qwen, llmjp = (pipeline.outputs[f].records for f in ('qwen3_8', 'llm_jp_4'))
        assert len(qwen) == len(llmjp) == 3, summary
        fields = ('source_qa_id', 'canonical_record_id', 'question', 'thinking', 'answer', 'canonical_reasoning_effort')
        for key in qwen:
            assert all(qwen[key][f] == llmjp[key][f] for f in fields)
        summary['provenance'] = 'Mock HTTP generation, real pinned official tokenizers; not actual LLM inference'
        summary['actual_llm_inference_calls'] = 0
        summary['mock_http_calls'] = generator.call_count
        summary['paired_fields_equal'] = list(fields)
        (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        (output / 'thinking_samples.md').write_text(
            '# Mock生成サンプル（実LLM生成ではありません）\n\n'
            '構造検証用token範囲: 全effort 1〜4096。通常設定のeffort別範囲の実証ではありません。\n\n'
            + '\n\n'.join(f'# {e}\n\n{t}' for e, t in FIXTURES.items()) + '\n')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        await pipeline.aclose()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer-cache', required=True)
    parser.add_argument('--output', type=Path, default=Path('/tmp/reasoning-effort-mock-sample'))
    args = parser.parse_args()
    asyncio.run(main(args.tokenizer_cache, args.output))
