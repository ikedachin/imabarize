# メイン処理：Wikipedia抽出・CPT・Q&A生成

[トップ](../../README.md) · [メイン処理](../main/README.md) · [アップロード](../upload/README.md) · [特殊なデータセット](../special_datasets/README.md)

以下のコマンドはすべてリポジトリルートで実行します。先に[共通セットアップ](../../README.md#セットアップ)を行ってください。入力例はローカルに用意するデータ、設定例は実行環境に合わせて編集するものです。

## 目的と処理の流れ

Wikipedia等の原文から、継続事前学習用の本文（CPT）や今治弁のQ&Aを作ります。

| 処理 | 実行ファイル | 入力 | 主な出力 |
| --- | --- | --- | --- |
| Wikipedia抽出 | `main_extract_wiki.py` | XML / XML.BZ2 | `id`・`title`・`content` のJSONL |
| CPT生成 | `main_create_cpt_dataset.py` | 原文のJSONL・テキスト等 | `all.jsonl`、`train.jsonl`、`validation.jsonl` 等 |
| Q&A生成 | `main_create_imabari_qa.py` | 原文またはCPTのJSONL等 | `question`・`thinking`・`answer`・`messages` 等のJSONL |

基本の流れは「Wikipedia抽出 → CPT生成 → Q&A生成」です。既存の原文やJaQuADから直接Q&Aを生成することもできます。QAの本文キーは原文なら `content`、CPTなら `text`、JaQuADなら `context` を指定します。

設定テンプレートをローカル用にコピーし、`output_path`、モデル、API接続先、プロンプトを編集します。以下の `cp -n` は既存設定を上書きしません。`openrouter: true` なら `openrouter_api_key` / `openrouter_server_url` / `openrouter_model_name`、`false` なら `SERVER_URL` / `MODEL_NAME` を設定します。この系統はAPIキーをYAMLから読み取ります。

```bash
cp -n yamls/create_cpt_dataset_settings_format.yaml yamls/create_cpt_dataset_settings.yaml
cp -n yamls/create_imabari_qa_settings_format.yaml yamls/create_imabari_qa_settings.yaml

uv run python main_create_cpt_dataset.py \
  -s ./test_source/wiki/raw.jsonl \
  -p ./yamls/create_cpt_dataset_settings.yaml

uv run python main_create_imabari_qa.py \
  -s ./test_output/cpt/wiki/all.jsonl \
  -p ./yamls/create_imabari_qa_settings.yaml \
  -t text
```

CPTの出力先を変更した場合は、次のQ&Aの `-s` も合わせてください。評価・GRPO・Reasoning Effortへの派生は[特殊なデータセット](../special_datasets/README.md)を参照してください。

## Wikipedia XML 抽出（`main_extract_wiki.py`）

Wikipedia の XML ダンプから、タイトルまたは本文に `今治` を含む一般記事を抽出し、CPT 生成などで使いやすい JSONL に保存します。非圧縮 XML と `.bz2` 圧縮済み XML の両方に対応しています。

実行例:

```bash
uv run python main_extract_wiki.py \
  --input ./wiki/jawiki-2026-05-01-p1p2391393.xml.bz2 \
  --output ./test_source/wiki/raw.jsonl \
  --content-threshold 3
```

主なCLI引数:

- `-i, --input`: Wikipedia XML ダンプのパス（デフォルト: `test_source/wiki/jawiki-****-**-**-**********.xml`）
- `-o, --output`: 出力 JSONL ファイルのパス（デフォルト: `test_source/wiki/raw.jsonl`）
- `-t, --content-threshold`: 本文に `今治` が何回以上出現したら抽出対象にするか（デフォルト: `3`）

抽出条件:

- namespace `0` の一般記事のみを対象にします。
- リダイレクト記事は除外します。
- タイトルに `今治` を含む記事は抽出します。
- タイトルに含まれない場合でも、本文中の `今治` の出現回数が `content-threshold` 以上なら抽出します。
- 脚注、外部リンク、テンプレート、表、画像リンクなどは可能な範囲で除去し、本文をプレーンテキスト化します。

## Q&A 生成

設定テンプレートをコピーして編集:

```bash
cp -n yamls/create_imabari_qa_settings_format.yaml yamls/create_imabari_qa_settings.yaml
```

`asyncio.Queue` に入力 item を積み、`asyncio.create_task` で起動した複数 worker が item ごとに Q&A 生成の各 step を進める Queue / worker pool 方式です。Step 単位で全件完了を待つ同期バリアを置かず、処理できる item から逐次進みます。

以下の2ファイルで動作します。

- `main_create_imabari_qa.py`
- `pipelines/create_qa_model.py`

実行例:

```bash
uv run python main_create_imabari_qa.py \
  -s ./test_source/JaQuAD_jsonls/validation.jsonl \
  -p ./yamls/create_imabari_qa_settings.yaml \
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

Queue / worker pool 版で追加利用できる主な設定:

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
- `pipeline_batch_size`: 入力処理窓の目安です。ログ上 `input_window_hint` として表示されます。
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

## CPT データセット生成

`test_source/wiki/raw.jsonl` の `content` を使って、継続事前学習向けの `train.jsonl` / `validation.jsonl` を作ります。`copyright_mitigation: true` の場合は、OpenAI 互換 API で「箇条書き化 → 再文章化」を行ってから保存します。

`asyncio.Queue` にCPTチャンク候補を積み、`asyncio.create_task` で起動した複数workerが空き次第「箇条書き化 → 再文章化 → 保存用chunk作成」を進める Queue / worker pool 方式です。`max_in_flight` はパイプライン全体で同時に vLLM / OpenAI 互換 API へ投げてよい最大リクエスト数です。

```bash
uv run python main_create_cpt_dataset.py \
  -s ./test_source/wiki/raw.jsonl \
  -p ./yamls/create_cpt_dataset_settings.yaml
```

主な特徴:

- `copyright_mitigation: true` の場合だけHTTPリクエストを行います。
- `copyright_mitigation: false` ではworker pool経由でもHTTPなしで通常のCPT chunkを作ります。
- 候補単位の失敗は全体を止めず、`all.failures.jsonl` に `id` / `chunk_index` / `failed_step` / `error` を保存します。
- 失敗候補があるentryはcache済みにしないため、再実行時に未完了entryを処理できます。

出力先は YAML の `output_path` で指定します。デフォルトでは以下に保存されます。

```text
test_output/cpt/wiki/all.jsonl
test_output/cpt/wiki/all.failures.jsonl
test_output/cpt/wiki/batch_status.jsonl
test_output/cpt/wiki/cache_processed_ids.txt
test_output/cpt/wiki/train.jsonl
test_output/cpt/wiki/validation.jsonl
test_output/cpt/wiki/stats.json
```

主な設定:

- `target_key`: CPT 本文に使う入力キー（Wiki データでは `content`）
- `include_title`: `title` を本文の先頭に付けるか
- `min_chars` / `max_chars` / `overlap_chars`: チャンク化の文字数設定。`copyright_mitigation: true` の場合は1リクエストが重くなるため、Qwen3系のローカルvLLMでは `max_chars: 512` 程度から確認します。
- `copyright_mitigation`: 版権対策の再構成処理を使うか
- `copyright_mitigation_failure_policy`: `original` なら版権対策再構成に失敗したchunkを元テキストで保存して完走を優先します。`fail` なら `.failures.jsonl` に保存します。
- `prompts`: 箇条書き化・再文章化プロンプト
- `batch_size`: API 推論の並列数
- `max_in_flight`: 非同期版で同時送信する最大リクエスト数。未指定時は `batch_size` を使います。
- `pipeline_batch_size`: 非同期版の入力処理窓の目安です。ログ上 `input_window_hint` として表示されます。
- `max_connections` / `max_keepalive_connections`: httpx の接続プール設定です。
- `read_timeout`: 1リクエストの応答待ち上限です。パイプライン全体の制限時間ではありません。
- `connect_timeout` / `pool_timeout`: 接続確立と接続プール待ちの timeout です。
- `keepalive_expiry`: keep-alive 接続を維持する秒数です。
- `http2`: HTTP/2 を使うかどうかです。vLLM の OpenAI 互換サーバーでは HTTP/1.1 のまま安定することが多いため、デフォルトは `false` です。
- `cpt_enable_thinking`: Qwen3 などの thinking 対応モデルに `chat_template_kwargs.enable_thinking` を送るかどうかです。`false` でCPT生成時のthinkingを無効化し、未指定ならサーバー側の既定値を使います。
- `train_ratio`: train 分割比率
- `text_key`: 出力 JSONL の本文キー（通常は `text`）

## 入出力フォーマット

### 入力（JSON / JSONL）

各レコードは辞書形式。`target_key` で指定したキーを変換対象として使用します。
`target_key` 未指定時は `text` または `content` を探索します。

例:

```json
{"book":"sample_book","page":1,"context":"これはテストです。"}
```

### 出力（JSONL）

Q&A 生成（`main_create_imabari_qa.py`）では、`question` / `thinking` / `answer` などのキーを持つ JSONL が出力されます。生成に失敗した item は、同名の `.failures.jsonl` に `failed_step` / `error` / `previous_outputs` などを保存します。

Wikipedia 抽出（`main_extract_wiki.py`）では、以下のように `id` / `title` / `content` を持つ JSONL が出力されます。

```json
{"id":"371","title":"今治市","content":"今治市は、愛媛県の北東部に位置する市..."}
```

CPT データセット生成（`main_create_cpt_dataset.py`）では、以下のように `text` とメタデータを持つ JSONL が出力されます。

```json
{"text":"記事タイトル\n\n本文...", "id":"371", "title":"愛媛県", "source_file":"...", "chunk_index":0}
```

GRPO 向け4択 Q&A 生成（`main_create_grpo_qa.py`）では、以下のように問題、4択、正解、無参照回答、参照あり回答、適性判定を持つ JSONL が出力されます。

```json
{"id":"371","chunk_index":0,"title":"愛媛県","question":"...","choices":[{"label":"A","text":"..."},{"label":"B","text":"..."},{"label":"C","text":"..."},{"label":"D","text":"..."}],"correct_label":"A","correct_answer":"...","blind_answer":"...","grounded_answer":"...","evidence":"...","difficulty":"borderline","rl_suitability":"accepted","rejection_reason":"","qa_generator":"Qwen3-30B-A3B-Instruct-2507","messages":[{"role":"user","content":"..."},{"role":"assistant","content":"A. ..."}]}
```

## 再実行時のスキップ仕様

`main_create_imabari_qa.py` はキャッシュファイルを使って `id` 単位で重複処理を避けます。`id` に加えて `chunk_index` もキャッシュキーに含められるため、同じ `id` の複数チャンクを個別に扱えます。
