# Imabarize Repository
![](./imabarize.png)
画像はnanobananaです

※このリポジトリは`https://github.com/foxn2000/sdg`にインスパイアされたレポジトリです。sdgレポジトリはさらに進化し、`foxn2000
sdg_loom`に進化しています。こちらも是非ともご覧ください。


このリポジトリは主に以下の処理を行います。
- `main_create_imabari_qa.py`: テキストや JSON/JSONL から Q&A データを生成
- `main_create_imabari_qa_httpx.py`: vLLM などの OpenAI 互換サーバー向けに、httpx 非同期リクエストで Q&A データを高速生成
- `main_create_imabari_qa_httpx_pipeline_pool.py`: `asyncio.Queue` と `asyncio.create_task` による worker pool 方式で Q&A データを逐次生成
- `main_create_cpt_dataset.py`: テキストや JSON/JSONL から継続事前学習（CPT）用データセットを生成
- `main_upload_cpt_dataset.py`: 生成済み CPT データセットを Hugging Face Hub にアップロード
- `main_extract_wiki.py`: Wikipedia XML ダンプから特定キーワードを含む記事を抽出し JSONL に保存

Q&A 生成、今治弁変換、CPT データセット生成は OpenAI 互換 API（OpenRouter またはローカルサーバー）を利用できます。CPT データセット生成では、必要に応じて入力テキストを箇条書き化してから再度文章化し、文章を再構成した JSONL を保存します。Wikipedia 抽出は API を使わず、XML または `.bz2` 圧縮済み XML を直接パースします。

## 主な機能

- JSON / JSONL / テキスト入力のバッチ処理
- Wikipedia XML / XML.BZ2 ダンプからのキーワード記事抽出
- `target_key` 指定による対象キーの切り替え
- CPT 用の本文正規化・チャンク化・版権対策再構成・train/validation 分割
- バッチ推論（`batch_size`）
- `httpx.AsyncClient` による非同期 Q&A 生成
- `asyncio.Queue` と `asyncio.create_task` による worker pool 型 Q&A パイプライン
- `max_in_flight` による vLLM / OpenAI 互換 API への同時リクエスト数制御
- 生成結果の到着順保存と、失敗レコードの `.failures.jsonl` 保存
- 既処理データのスキップ（`book` + `page` または `id` キャッシュ）
- OpenRouter / ローカル OpenAI 互換 API の切り替え
- 一部作成者の都合により使っていない機能があります
- 実にくだらない、でも私にとって満足感の高いリポジトリです

## リポジトリ構成

- `main_create_imabari_qa.py`: Q&A 生成の実行スクリプト
- `main_create_imabari_qa_httpx.py`: Q&A 生成の httpx 非同期版実行スクリプト
- `main_create_imabari_qa_httpx_pipeline_pool.py`: Q&A 生成の Queue / worker pool 非同期版実行スクリプト
- `main_create_cpt_dataset.py`: CPT データセット生成の実行スクリプト
- `main_upload_cpt_dataset.py`: CPT データセットの Hugging Face Hub アップロードスクリプト
- `main_extract_wiki.py`: Wikipedia XML ダンプから今治関連記事を抽出する実行スクリプト
- `pipelines/imabarize_pipeline.py`: 今治弁変換の推論・保存処理
- `pipelines/create_qa_model.py`: Q&A 生成の推論処理
- `pipelines/create_qa_model_httpx.py`: Q&A 生成の httpx 非同期推論処理
- `pipelines/create_qa_model_httpx_pipeline_pool.py`: Queue / worker pool 方式の httpx 非同期 Q&A 推論処理
- `pipelines/create_cpt_dataset.py`: CPT 用の正規化・チャンク化・保存処理
- `prompts/imabarize.md`: 今治弁変換プロンプト
- `prompts/create_qa/`: Q&A 生成プロンプト群
- `prompts/create_cpt/`: CPT 版権対策用プロンプト群
- `yamls/imabari_settings_format.yaml`: Q&A 生成向け設定テンプレート
- `yamls/cpt_wiki_settings_format.yaml`: CPT 生成向け設定テンプレート
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

### A. Q&A 生成

設定テンプレートをコピーして編集:

```bash
cp yamls/imabari_settings_format.yaml yamls/imabari_settings.yaml
```

#### A-1. 同期版（`main_create_imabari_qa.py`）

既存の同期版です。シンプルな処理確認や低並列での生成に使います。

```bash
python main_create_imabari_qa.py \
  -s ./test_source/JaQuAD_jsonls/validation.jsonl \
  -p ./yamls/imabari_settings.yaml \
  -t context
```

#### A-2. httpx 非同期版（`main_create_imabari_qa_httpx.py`）

vLLM などのローカル OpenAI 互換サーバーを高負荷で回したい場合は、httpx 非同期版を使えます。既存の同期版ファイルは残したまま、以下の2ファイルで動作します。

- `main_create_imabari_qa_httpx.py`
- `pipelines/create_qa_model_httpx.py`

実行例:

```bash
python main_create_imabari_qa_httpx.py \
  -s ./test_source/JaQuAD_jsonls/validation.jsonl \
  -p ./yamls/imabari_settings.yaml \
  -t context
```

#### A-3. Queue / worker pool 非同期版（`main_create_imabari_qa_httpx_pipeline_pool.py`）

`asyncio.Queue` に入力 item を積み、`asyncio.create_task` で起動した複数 worker が item ごとに Q&A 生成の各 step を進める版です。Step 単位で全件完了を待つ同期バリアを置かず、処理できる item から逐次進みます。

以下の2ファイルで動作します。

- `main_create_imabari_qa_httpx_pipeline_pool.py`
- `pipelines/create_qa_model_httpx_pipeline_pool.py`

実行例:

```bash
python main_create_imabari_qa_httpx_pipeline_pool.py \
  -s ./test_source/JaQuAD_jsonls/validation.jsonl \
  -p ./yamls/imabari_settings.yaml \
  -t context
```

主な特徴:

- `asyncio.Queue` に item id を投入し、worker が item 単位で step 1 から step 5 まで処理します。
- worker は `asyncio.create_task` で起動されます。
- worker 数は `min(max_in_flight, 入力件数)` で決まります。
- `max_in_flight` はパイプライン全体で同時に vLLM / OpenAI 互換 API へ投げてよい最大リクエスト数です。
- 結果は `on_result` callback で到着次第 JSONL に追記されます。
- 失敗した item は処理全体を止めず、`.failures.jsonl` に保存されます。
- JSON/JSONL 入力では `id` と `chunk_index` の組み合わせをキャッシュキーにできるため、CPT チャンク由来の入力も再実行しやすくなっています。

httpx 非同期版と Queue / worker pool 非同期版で追加利用できる主な設定:

```yaml
batch_size: 8
max_in_flight: 8
pipeline_batch_size: 32
max_connections: 16
max_keepalive_connections: 8
connect_timeout: 5
pool_timeout: 30
keepalive_expiry: 120
http2: false
```

- `max_in_flight`: vLLM サーバーに同時送信する最大リクエスト数。GPU使用率を見ながら調整します。
- `pipeline_batch_size`: 入力処理窓の目安です。Queue / worker pool 版ではログ上 `input_window_hint` として表示されます。
- `max_connections` / `max_keepalive_connections`: httpx の接続プール設定です。基本は `max_in_flight` 以上にします。
- `read_timeout`: 1リクエストの応答待ち上限です。パイプライン全体の制限時間ではありません。
- `connect_timeout` / `pool_timeout`: 接続確立と接続プール待ちの timeout です。
- `http2`: vLLM の OpenAI 互換サーバーでは HTTP/1.1 のまま安定することが多いため、デフォルトは `false` です。

`thinking_enabled_by_step` を設定すると、step ごとに `chat_template_kwargs.enable_thinking` を切り替えられます。

```yaml
thinking_enabled_by_step:
  question: true
  answer: true
  thinking: true
  refine_answer: false
  eval: false
```

### B. CPT データセット生成（`main_create_cpt_dataset.py`）

`test_source/wiki/raw.jsonl` の `content` を使って、継続事前学習向けの `train.jsonl` / `validation.jsonl` を作ります。`copyright_mitigation: true` の場合は、OpenAI 互換 API で「箇条書き化 → 再文章化」を行ってから保存します。

```bash
python main_create_cpt_dataset.py \
  -s ./test_source/wiki/raw.jsonl \
  -p ./yamls/cpt_wiki_settings_format.yaml
```

出力先は YAML の `output_path` で指定します。デフォルトでは以下に保存されます。

```text
test_output/cpt/wiki/all.jsonl
test_output/cpt/wiki/batch_status.jsonl
test_output/cpt/wiki/cache_processed_ids.txt
test_output/cpt/wiki/train.jsonl
test_output/cpt/wiki/validation.jsonl
test_output/cpt/wiki/stats.json
```

主な設定:

- `target_key`: CPT 本文に使う入力キー（Wiki データでは `content`）
- `include_title`: `title` を本文の先頭に付けるか
- `min_chars` / `max_chars` / `overlap_chars`: チャンク化の文字数設定
- `copyright_mitigation`: 版権対策の再構成処理を使うか
- `prompts`: 箇条書き化・再文章化プロンプト
- `batch_size`: API 推論の並列数
- `train_ratio`: train 分割比率
- `text_key`: 出力 JSONL の本文キー（通常は `text`）

### C. CPT データセットのアップロード（`main_upload_cpt_dataset.py`）

`main_create_cpt_dataset.py` で生成した CPT データセットを Hugging Face Hub の dataset repository にアップロードします。デフォルトでは `all.jsonl` を canonical なアップロード対象にし、`--include-splits` を付けた場合だけ `train.jsonl` / `validation.jsonl` もアップロードします。

dry-run:

```bash
python main_upload_cpt_dataset.py \
  --repo-id YOUR_NAME/YOUR_DATASET \
  --settings-path ./yamls/cpt_wiki_settings_format.yaml \
  --dry-run
```

アップロード:

```bash
python main_upload_cpt_dataset.py \
  --repo-id YOUR_NAME/YOUR_DATASET \
  --hf-token YOUR_HF_TOKEN \
  --settings-path ./yamls/cpt_wiki_settings_format.yaml
```

### D. Wikipedia XML 抽出（`main_extract_wiki.py`）

Wikipedia の XML ダンプから、タイトルまたは本文に `今治` を含む一般記事を抽出し、CPT 生成などで使いやすい JSONL に保存します。非圧縮 XML と `.bz2` 圧縮済み XML の両方に対応しています。

実行例:

```bash
python main_extract_wiki.py \
  --input ./wiki/jawiki-2026-05-01-p1p2391393.xml.bz2 \
  --output ./test_source/wiki/raw.jsonl \
  --content-threshold 3
```

主なCLI引数:

- `-i, --input`: Wikipedia XML ダンプのパス（デフォルト: `wiki/jawiki-2026-05-01-p1p2391393.xml`）
- `-o, --output`: 出力 JSONL ファイルのパス（デフォルト: `data/imabari/raw.jsonl`）
- `-t, --content-threshold`: 本文に `今治` が何回以上出現したら抽出対象にするか（デフォルト: `3`）

抽出条件:

- namespace `0` の一般記事のみを対象にします。
- リダイレクト記事は除外します。
- タイトルに `今治` を含む記事は抽出します。
- タイトルに含まれない場合でも、本文中の `今治` の出現回数が `content-threshold` 以上なら抽出します。
- 脚注、外部リンク、テンプレート、表、画像リンクなどは可能な範囲で除去し、本文をプレーンテキスト化します。

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
`main_create_imabari_qa_httpx.py` と `main_create_imabari_qa_httpx_pipeline_pool.py` も同じ形式を出力します。生成に失敗した item は、同名の `.failures.jsonl` に `failed_step` / `error` / `previous_outputs` などを保存します。

Wikipedia 抽出（`main_extract_wiki.py`）では、以下のように `id` / `title` / `content` を持つ JSONL が出力されます。

```json
{"id":"371","title":"今治市","content":"今治市は、愛媛県の北東部に位置する市..."}
```

CPT データセット生成（`main_create_cpt_dataset.py`）では、以下のように `text` とメタデータを持つ JSONL が出力されます。

```json
{"text":"記事タイトル\n\n本文...", "id":"371", "title":"愛媛県", "source_file":"...", "chunk_index":0}
```

## 再実行時のスキップ仕様

`main_create_imabari_qa.py` と `main_create_imabari_qa_httpx.py` はキャッシュファイルを使って `id` 単位で重複処理を避けます。
`main_create_imabari_qa_httpx_pipeline_pool.py` は `id` に加えて `chunk_index` もキャッシュキーに含められるため、同じ `id` の複数チャンクを個別に扱えます。

## ライセンス
Apache License 2.0です。
`LICENSE` を参照してください。
