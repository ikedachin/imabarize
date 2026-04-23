# Imabarize Repository
![](./imabarize.png)
画像はnanobananaです

※このリポジトリは`https://github.com/foxn2000/sdg`にインスパイアされたレポジトリです。sdgレポジトリはさらに進化し、`foxn2000
sdg_loom`に進化しています。こちらも是非ともご覧ください。


このリポジトリは主に以下の処理を行います。
- `main_create_imabari_qa.py`: テキストや JSON/JSONL から Q&A データを生成

どちらも OpenAI 互換 API（OpenRouter またはローカルサーバー）を利用し、結果を JSONL 形式で保存します。

## 主な機能

- JSON / JSONL / テキスト入力のバッチ処理
- `target_key` 指定による対象キーの切り替え
- バッチ推論（`batch_size`）
- 既処理データのスキップ（`book` + `page` または `id` キャッシュ）
- OpenRouter / ローカル OpenAI 互換 API の切り替え
- 一部作成者の都合により使っていない機能があります
- 実にくだらない、でも私にとって満足感の高いリポジトリです

## リポジトリ構成

- `main_create_imabari_qa.py`: Q&A 生成の実行スクリプト
- `pipelines/imabarize_pipeline.py`: 今治弁変換の推論・保存処理
- `pipelines/create_qa_model.py`: Q&A 生成の推論処理
- `prompts/imabarize.md`: 今治弁変換プロンプト
- `prompts/create_qa/`: Q&A 生成プロンプト群
- `yamls/imabari_settings_format.yaml`: Q&A 生成向け設定テンプレート
- `test_source/`: 入力サンプル
- `test_output/`: 出力先サンプル

## 成果物の例
[JaQuAD_imabari_v1](https://huggingface.co/datasets/ikedachin/JaQuAD_imabari_v1)  
[JaQuAD_imabari_v2](https://huggingface.co/datasets/ikedachin/JaQuAD_imabari_v2)


## セットアップ

前提:

- Python 3.11+
- OpenAI互換 Chat Completions API を提供するエンドポイント

### uv（推奨）

```bash
uv sync
```

### venv + pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 実行方法

### A. 今治弁変換（`main_imabarize.py`）

### 1) 設定ファイルを用意

`main_imabarize.py` には `imabarize_prompt` を含む設定が必要です。
現在の `yamls/imabari_settings_format.yaml` は Q&A 生成向けのため、今治弁変換ではそのまま利用できません。

例: `yamls/imabarize_only.yaml`

```yaml
openrouter: true
openrouter_api_key: "YOUR_API_KEY"
openrouter_server_url: "https://openrouter.ai/api/v1"
openrouter_model_name: "qwen/qwen3.5-27b"

SERVER_URL: "http://localhost:8000/v1"
MODEL_NAME: "Qwen3-30B-A3B-Instruct-2507"

infer_config:
  max_tokens: 4096
  temperature: 0
  top_p: 1.0

batch_size: 8
prompts:
  - imabarize_prompt: ./prompts/imabarize.md
output_path: ./test_output/imabarized
max_retries: 3
wait_seconds: 5
```

### 2) 実行

ディレクトリ入力:

```bash
python main_imabarize.py \
  -s ./test_source/dummy \
  -p ./yamls/imabarize_only.yaml \
  -t context \
  -e jsonl
```

単一ファイル入力:

```bash
python main_imabarize.py \
  -s ./test_source/dummy/dummy.jsonl \
  -p ./yamls/imabarize_only.yaml \
  -t context
```

主なCLI引数:

- `-s, --source`: 入力ファイルまたはディレクトリ
- `-p, --settings_path`: YAML設定ファイル
- `-t, --target_key`: 変換対象キー（未指定時は `text`）
- `-e, --extensions`: 対象拡張子（例: `json,jsonl`）
- `-i, --start_index`: 将来の再開処理向け引数（現状は実処理には未反映）

### B. Q&A 生成（`main_create_imabari_qa.py`）

設定テンプレートをコピーして編集:

```bash
cp yamls/imabari_settings_format.yaml yamls/imabari_settings.yaml
```

実行例:

```bash
python main_create_imabari_qa.py \
  -s ./test_source/JaQuAD_jsonls/validation.jsonl \
  -p ./yamls/imabari_settings.yaml \
  -t context
```

## 入出力フォーマット

### 入力（JSON / JSONL）

各レコードは辞書形式。`target_key` で指定したキーを変換対象として使用します。  
`target_key` 未指定時は `text` または `content` を探索します。

例:

```json
{"book":"sample_book","page":1,"context":"これはテストです。"}
```

### 出力（JSONL）

Q&A 生成（`main_create_imabari_qa.py`）では、`question` / `thinking` / `answer` などのキーを持つ JSONL が出力されます。

## 再実行時のスキップ仕様

`main_imabarize.py` は `output_path` 配下の既存 JSONL を読み、`book` と `page` が一致する入力レコードをスキップします。  
`main_create_imabari_qa.py` はキャッシュファイルを使って `id` 単位で重複処理を避けます。

## ライセンス
Apache License 2.0です。
`LICENSE` を参照してください。


