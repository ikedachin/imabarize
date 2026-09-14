# Imabarize Repository
![](./imabarize.png)
画像はnanobananaです

※このリポジトリは`https://github.com/foxn2000/sdg`にインスパイアされたレポジトリです。sdgレポジトリはさらに進化し、`foxn2000
sdg_loom`に進化しています。こちらも是非ともご覧ください。


このリポジトリは主に以下の処理を行います。
- `main_create_imabari_qa.py`: `asyncio.Queue` と `asyncio.create_task` による worker pool 方式で Q&A データを逐次生成
- `main_create_eval_qa.py`: worker pool 方式で評価用 Q&A データセットを生成
- `main_judge_eval_qa.py`: 評価用 Q&A と回答 JSONL を外部 LLM で LLM-as-a-Judge 採点
- `main_create_cpt_dataset.py`: `asyncio.Queue` と `asyncio.create_task` による worker pool 方式で CPT データセットを逐次生成
- `main_create_grpo_qa.py`: CPT 生成済み JSONL から GRPO 向け4択 Q&A データセットを生成
- `main_create_reasoning_effort_dataset.py`: QAの質問・回答を固定し、effort別thinkingとQwen／llm-jp向けSFTデータセットを生成
- `main_upload_dataset.py`: 生成済み CPT データセットを Hugging Face Hub にアップロード
- `main_extract_wiki.py`: Wikipedia XML ダンプから特定キーワードを含む記事を抽出し JSONL に保存

Q&A 生成、今治弁変換、CPT データセット生成は OpenAI 互換 API（OpenRouter またはローカルサーバー）を利用できます。CPT データセット生成では、必要に応じて入力テキストを箇条書き化してから再度文章化し、文章を再構成した JSONL を保存します。Wikipedia 抽出は API を使わず、XML または `.bz2` 圧縮済み XML を直接パースします。

## 主な機能

- JSON / JSONL / テキスト入力からの Q&A / CPT データセット生成
- Wikipedia XML / XML.BZ2 ダンプからのキーワード記事抽出
- `target_key` 指定による対象キーの切り替え
- CPT 用の本文正規化・チャンク化・版権対策再構成・train/validation 分割
- `asyncio.Queue` と `asyncio.create_task` による worker pool 型 Q&A / CPT パイプライン
- `max_in_flight` による vLLM / OpenAI 互換 API への同時リクエスト数制御
- 生成結果の到着順保存と、失敗レコードの `.failures.jsonl` 保存
- 評価用 Q&A データセット生成と LLM-as-a-Judge 採点
- GRPO / RL 用の4択 Q&A データセット生成
- 既処理データのスキップ（`book` + `page` または `id` キャッシュ）
- OpenRouter / ローカル OpenAI 互換 API の切り替え
- 一部作成者の都合により使っていない機能があります
- 実にくだらない、でも私にとって満足感の高いリポジトリです

## リポジトリ構成

- `main_create_imabari_qa.py`: Q&A 生成の Queue / worker pool 非同期版実行スクリプト
- `main_create_eval_qa.py`: 評価用 Q&A 生成の Queue / worker pool 非同期版実行スクリプト
- `main_judge_eval_qa.py`: 評価用 Q&A の LLM-as-a-Judge 実行スクリプト
- `main_create_cpt_dataset.py`: CPT データセット生成の Queue / worker pool 非同期版実行スクリプト
- `main_create_grpo_qa.py`: GRPO 向け4択 Q&A 生成の Queue / worker pool 非同期版実行スクリプト
- `main_create_reasoning_effort_dataset.py`: Reasoning Effort生成CLI
- `reasoning_effort/`: 共通thinkingの生成・検証、モデル別整形、キャッシュ・出力スキーマ・進捗表示
- `migrate_reasoning_effort_output.py`: 単一の旧出力JSONLを新形式とresume情報へ変換
- `migrate_reasoning_effort_directory.py`: 両モデル出力・キャッシュ・resume情報・失敗履歴をディレクトリ単位で移行・統合
- `main_upload_dataset.py`: CPT データセットの Hugging Face Hub アップロードスクリプト
- `main_extract_wiki.py`: Wikipedia XML ダンプから今治関連記事を抽出する実行スクリプト
- `pipelines/imabarize_pipeline.py`: 今治弁変換の推論・保存処理
- `pipelines/create_qa_model.py`: Queue / worker pool 方式の httpx 非同期 Q&A 推論処理
- `pipelines/judge_eval_qa.py`: Queue / worker pool 方式の LLM-as-a-Judge 採点処理
- `pipelines/create_cpt_dataset.py`: Queue / worker pool 方式の httpx 非同期 CPT 生成処理
- `pipelines/create_rl_qa.py`: Queue / worker pool 方式の httpx 非同期 GRPO 4択 Q&A 生成処理
- `prompts/imabarize.md`: 今治弁変換プロンプト
- `prompts/create_qa/`: Q&A 生成プロンプト群
- `prompts/judge_eval_qa/`: LLM-as-a-Judge 採点プロンプト群
- `prompts/create_cpt/`: CPT 版権対策用プロンプト群
- `prompts/create_rl_qa/`: GRPO 4択 Q&A 生成プロンプト群
- `yamls/create_imabari_qa_settings_format.yaml`: Q&A 生成（`main_create_imabari_qa.py`）向け設定テンプレート
- `yamls/create_eval_qa_settings_format.yaml`: 評価用 Q&A 生成（`main_create_eval_qa.py`）向け設定テンプレート
- `yamls/judge_eval_qa_settings_format.yaml`: LLM-as-a-Judge 採点（`main_judge_eval_qa.py`）向け設定テンプレート
- `yamls/create_cpt_dataset_settings_format.yaml`: CPT 生成（`main_create_cpt_dataset.py`）向け設定テンプレート
- `yamls/create_grpo_qa_settings_format.yaml`: GRPO 4択 Q&A 生成（`main_create_grpo_qa.py`）向け設定テンプレート
- `yamls/upload_dataset_settings_format.yaml`: データセットアップロード（`main_upload_dataset.py`）向け設定テンプレート
- `test_source/`: 入力サンプル
- `test_output/`: 出力先サンプル

## 成果物の例
### CPT set
[imabari_wiki_cpt_v3](https://huggingface.co/datasets/ikedachin/imabari_wiki_cpt_v3)  


### QA set
[JaQuAD_imabari_v1](https://huggingface.co/datasets/ikedachin/JaQuAD_imabari_v1)  
[JaQuAD_imabari_v2](https://huggingface.co/datasets/ikedachin/JaQuAD_imabari_v2)  
[imabari_wiki_qa_v3](https://huggingface.co/datasets/ikedachin/imabari_wiki_qa_v3)  
[imabari_wiki_qa_v4](https://huggingface.co/datasets/ikedachin/imabari_wiki_qa_v4)  



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
cp yamls/create_imabari_qa_settings_format.yaml yamls/create_imabari_qa_settings.yaml
```

`asyncio.Queue` に入力 item を積み、`asyncio.create_task` で起動した複数 worker が item ごとに Q&A 生成の各 step を進める Queue / worker pool 方式です。Step 単位で全件完了を待つ同期バリアを置かず、処理できる item から逐次進みます。

以下の2ファイルで動作します。

- `main_create_imabari_qa.py`
- `pipelines/create_qa_model.py`

実行例:

```bash
python main_create_imabari_qa.py \
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

### B. 評価用 Q&A データセット生成

評価用 Q&A は既存の Q&A JSONL と同じ形式で出力します。実装は `main_create_imabari_qa.py` と同じ worker pool 型の生成処理を使い、評価用途の設定テンプレートを分けています。

設定テンプレートをコピーして編集:

```bash
cp yamls/create_eval_qa_settings_format.yaml yamls/create_eval_qa_settings.yaml
```

実行例:

```bash
python main_create_eval_qa.py \
  -s ./test_source/JaQuAD_jsonls/validation.jsonl \
  -p ./yamls/create_eval_qa_settings.yaml \
  -t context
```

作成件数は YAML の `sample_size` で指定できます。`sample_size: 100` なら未処理候補から最大100件を `seed` 固定でサンプリングします。CLI で一時的に上書きする場合は `-n` / `--sample_size` を使います。

```bash
python main_create_eval_qa.py \
  -s ./test_source/JaQuAD_jsonls/validation.jsonl \
  -p ./yamls/create_eval_qa_settings.yaml \
  -t context \
  -n 50
```

`sample_size` を `null` または 0 以下にすると全件処理します。出力先は YAML の `output_path` で指定します。出力レコードは `qa_id` / `id` / `chunk_index` / `source_files` / `question` / `thinking` / `answer` / `eval` / `qa_generator` / `messages` を持つ既存 Q&A 互換 JSONL です。`qa_id` はQ&Aレコードの生成時に割り当てる一意な UUIDv4 です。同じ `id + chunk_index` や同じ内容のレコードも、それぞれ異なる `qa_id` を持ちます。JSON/JSONL 入力では `id` と `chunk_index` を組み合わせた cache key で再実行済みレコードをスキップします。

### C. LLM-as-a-Judge 評価

`main_judge_eval_qa.py` は、評価用 Q&A JSONL と評価対象モデルの回答 JSONL を突合し、外部 LLM で採点します。このスクリプトは評価対象モデルの回答生成は行いません。

設定テンプレートをコピーして編集:

```bash
cp yamls/judge_eval_qa_settings_format.yaml yamls/judge_eval_qa_settings.yaml
```

評価対象モデルの回答 JSONL は、最低限以下のキーを持たせます。

```json
{"id":"371","chunk_index":0,"question":"...","answer":"評価対象モデルの回答"}
```

`question` は任意です。突合は `id` と `chunk_index` を優先し、`chunk_index` がないデータでは `id` 単位でも扱えます。回答本文のキー名を変えたい場合は YAML の `candidate_answer_key` を変更します。

実行例:

```bash
python main_judge_eval_qa.py \
  -q ./test_output/eval_qa/validation.jsonl \
  -a ./test_output/eval_answers/answers.jsonl \
  -p ./yamls/judge_eval_qa_settings.yaml
```

Judge の出力は YAML の `output_path` に保存されます。

- `all.jsonl`: `id` / `chunk_index` / `question` / `reference_answer` / `candidate_answer` / `judge_score` / `judge_label` / `judge_reason` / `judge_model`
- `all.failures.jsonl`: 採点に失敗したレコード。`failed_step` / `error` / `previous_outputs` を保存します。
- `cache_processed_ids.txt`: 再実行時に採点済みレコードをスキップする cache key。
- `stats.json`: 保存件数、失敗件数、ラベル別件数などの集計。

`judge_score` は 1 から 5 の整数、`judge_label` は `correct` / `partially_correct` / `incorrect` / `unjudgeable` のいずれかです。

### D. CPT データセット生成

`test_source/wiki/raw.jsonl` の `content` を使って、継続事前学習向けの `train.jsonl` / `validation.jsonl` を作ります。`copyright_mitigation: true` の場合は、OpenAI 互換 API で「箇条書き化 → 再文章化」を行ってから保存します。

`asyncio.Queue` にCPTチャンク候補を積み、`asyncio.create_task` で起動した複数workerが空き次第「箇条書き化 → 再文章化 → 保存用chunk作成」を進める Queue / worker pool 方式です。`max_in_flight` はパイプライン全体で同時に vLLM / OpenAI 互換 API へ投げてよい最大リクエスト数です。

```bash
python main_create_cpt_dataset.py \
  -s ./test_source/wiki/raw.jsonl \
  -p ./yamls/create_cpt_dataset_settings_format.yaml
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

### E. GRPO 向け4択 Q&A データセット生成

`test_output/cpt/wiki/all.jsonl` の `text` を参照情報として使い、GRPO / RL 用の4択 Q&A データセットを作ります。Q&A / CPT 生成と同じ Queue / worker pool 方式で、以下の2ファイルで動作します。

- `main_create_grpo_qa.py`
- `pipelines/create_rl_qa.py`

実行例:

```bash
python main_create_grpo_qa.py \
  -p ./yamls/create_grpo_qa_settings_format.yaml
```

入力を明示する場合:

```bash
python main_create_grpo_qa.py \
  -s ./test_output/cpt/wiki/all.jsonl \
  -p ./yamls/create_grpo_qa_settings_format.yaml
```

パイプラインは item ごとに以下の4 stepを順に実行します。

1. 参照情報をもとに標準語の問題文を作成
2. 参照情報なしで同じモデルに回答させる
3. 参照情報ありで正確な回答と根拠を作成
4. 無参照回答と参照あり回答を比較し、RL 用の4択選択肢と適性判定を作成

主な特徴:

- 有効な入力行から `seed` 固定で最大 `sample_size` 件をランダム抽出します。
- `asyncio.Queue` に item id を投入し、worker が item 単位で step 1 から step 4 まで処理します。
- worker 数は `min(max_in_flight, 入力件数)` で決まります。
- `max_in_flight` はパイプライン全体で同時に OpenAI 互換 API へ投げてよい最大リクエスト数です。
- `pipeline_batch_size` は入力処理窓の目安で、ログ上 `input_window_hint` として表示されます。
- 成功した item は `all.jsonl` に保存し、`rl_suitability == "accepted"` の行を学習対象として使えます。
- 失敗した item は `all.failures.jsonl` に `failed_step` / `error` / `previous_outputs` を保存します。
- 再実行時は `cache_processed_ids.txt` の `id + chunk_index` で成功済み item をスキップします。

デフォルト出力:

```text
test_output/rl_qa/wiki/all.jsonl
test_output/rl_qa/wiki/all.failures.jsonl
test_output/rl_qa/wiki/cache_processed_ids.txt
test_output/rl_qa/wiki/stats.json
```

主な設定:

- `source_path`: 入力 JSONL（デフォルトは `./test_output/cpt/wiki/all.jsonl`）
- `target_key`: 参照情報として使う本文キー（CPT 出力では `text`）
- `sample_size`: ランダム抽出する最大件数
- `seed`: ランダム抽出の固定 seed
- `prompts`: `prompts/create_rl_qa/` 配下の4 step 用プロンプト
- `thinking_enabled_by_step`: step ごとの `chat_template_kwargs.enable_thinking` 切り替え

### F. データセットのアップロード（`main_upload_dataset.py`）

生成済みの CPT / QA / GRPO データセットを Hugging Face Hub の dataset repository にアップロードします。デフォルトでは `all.jsonl` を canonical なアップロード対象にし、`--include-splits` を付けた場合だけ `train.jsonl` / `validation.jsonl` もアップロードします。

アップロード前に JSONL record から除外するキーは `--exclude-upload-key` または `--exclude-upload-keys` で指定できます。デフォルトでは `item_id` のみ除外します。

`--settings_path` は `output_path` を読み取るためだけに使われるため、`yamls/upload_dataset_settings_format.yaml` に加えて、生成時に使った `create_cpt_dataset_settings.yaml` / `create_imabari_qa_settings.yaml` / `create_grpo_qa_settings.yaml` などをそのまま指定してもアップロードできます。

dry-run:

```bash
python main_upload_dataset.py \
  --repo_id YOUR_NAME/YOUR_DATASET \
  --settings_path ./yamls/create_cpt_dataset_settings_format.yaml \
  --dry-run \
  --exclude-upload-key source_file \
  --exclude-upload-key copyright_mitigation
```

アップロード:

```bash
python main_upload_dataset.py \
  --repo_id YOUR_NAME/YOUR_DATASET \
  --hf_token YOUR_HF_TOKEN \
  --settings_path ./yamls/create_cpt_dataset_settings_format.yaml
```

### G. Wikipedia XML 抽出（`main_extract_wiki.py`）

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

## ライセンス
Apache License 2.0です。
`LICENSE` を参照してください。

## Reasoning Effort Dataset Generator

validated QA の question / answer を固定し、今治弁の thinking を canonical effort ごとに一度生成して、Qwen3.8 と llm-jp-4 向け SFT JSONL を同時に作ります。既存 pipeline とは独立した新規機能です。

| Canonical effort | 推論の方針 | Qwen3.8 | llm-jp-4 |
| --- | --- | --- | --- |
| low | 必要最小限の事実と推論 | low | low |
| medium | 標準的な分析・整理・確認 | medium | medium |
| high | 必要に応じた分解・比較・検証・整合性確認 | xhigh | high |

`expand_all` は1 QAあたり通常3回の推論で6レコードを作ります。モデル別の再推論はありません。形式・長さ・通信エラーの再試行が発生した場合は、実際の呼び出し数が増えます。`fixed` は指定した1 effort、`token_length` は指定 effort のプロンプトで1回生成し、reference tokenizer の実token数から low / medium / high を付与します。`token_length` の境界は隙間・重複なしで指定します。

```text
                question + answer (+ context)
                            │
                 Canonical low / medium / high
                            │
                      LLM inference
                            │
                 今治弁 thinking（共通）
                            │
               Format → Token Length validation
                            │
                    Canonical cache
                     ┌──────┴──────┐
                     ▼             ▼
                 Qwen3.8        llm-jp-4
                 Formatter      Formatter
               high → xhigh    high → high
                     │             │
               Chat template    Harmony validation
                     │             │
                qwen38.jsonl   llmjp4.jsonl
```

### 実行方法と設定

この機能には `httpx`, `transformers`, `datasets`, `jinja2`, `pyyaml`, `tqdm` が必要です。既存環境への依存追加は `pyproject.toml` に記載しています。リポジトリルートから実行してください。YAML内の相対パスも作業ディレクトリ基準です。

```bash
uv run python main_create_reasoning_effort_dataset.py \
  -p ./yamls/create_reasoning_effort_dataset.yaml
```

[設定テンプレート](yamls/create_reasoning_effort_dataset_settings_format.yaml)には全項目を記載しています。公開Tokenizerのrevisionは検証したコミットに固定しています。llm-jp は公式カスタムTokenizerを使用するため `trust_remote_code: true` です。生成モデルは出力対象Tokenizerとは独立して設定します。認証には `generator.api_key_env` が指定する環境変数を利用します（`.env` の自動読込はしません）。

```yaml
source:
  type: huggingface
  dataset_name: ikedachin/imabari_wiki_qa_v4_validated
  split: train
# ローカル入力の代替:
# source: {type: local, path: ./data/input.jsonl}
fields: {id: qa_id, question: question, answer: answer, thinking: thinking, context: null}
generator:
  provider: local
  server_url: http://localhost:8000/v1
  model_name: Qwen3.8-27B
  api_key_env: OPENAI_API_KEY
  generation:
    chat_template_kwargs: {enable_thinking: false}
    max_tokens: 4096
    temperature: 0.2
    top_p: 0.95
  use_reasoning_effort: false
  reasoning_effort_by_canonical: {low: low, medium: medium, high: xhigh}
reasoning_effort:
  mode: expand_all
```

この断片は全設定の代替ではありません。テンプレートの `thinking_generation`, `targets`, `output` 等と併せて指定します。provider は接続先の説明ラベルで、通信方式は共通の OpenAI互換 `/chat/completions` です。OpenRouterの場合はURLとモデル名・キー環境変数を設定してください。

### thinking の正規形

```text
## 思考プロセス

### 1. 見出し名

本文

### 2. 見出し名

本文
```

タイトル直後、Section Heading直後、Section本文と次Sectionの間には、**空行がちょうど1行**必要です。不要な連続空行は禁止です。番号は1から連続、最低2section、各sectionに本文が必要です。`<think>` / `</think>`、code fence、前置き・後書き、別のfinal answerを含めません。

`ThinkingFormatValidator` は行構造から機械可読エラー（例: `missing_blank_line_after_title`, `empty_section_body`）を返します。失敗時はエラーと修正指示を次のプロンプトへ渡します。`max_format_retries: 2` は初回と追加2回が上限です。長さの再試行は `max_length_retries` で別に管理します。検証を無効化する設定は受け付けません。CRLFと末尾改行1個だけを正規化し、壊れたMarkdownの自動修復は行いません。

今治弁は全effortで必須です。「〜けん」「〜とる」「〜よる」「〜なんよ」「ほうやけん」を文脈に応じて使い、語尾の機械的置換や過度な方言化を避けるプロンプトにしています。元の thinking は生成プロンプトへ渡しません。元answerに合わせた架空の根拠や、固有名詞・年・数値の創作も禁止しています。

### Token数とモデル別出力

`thinking_generation.length_reference_tokenizer` のtoken数を `canonical_thinking_tokens` に記録し、effort別の `min_tokens` / `max_tokens` と照合します。各Datasetの `thinking_tokens` はそれぞれの対象Tokenizerで計測します。特殊トークンは計数に追加しません。文字数による代替はありません。

Qwen / llm-jp の対応レコードは `source_qa_id`, `question`, `thinking`, `answer` を共有します。`qa_id` は内部canonical IDと対象familyから作ります。`source_metadata` は元QAの `qa_id`, `id`, `chunk_index` の３項目だけです。`original_thinking` は最終Datasetには保存しません。`keep_original_thinking` は内部レコードの保持設定として維持します。

最終Datasetのモデル差は `reasoning_effort`, `thinking_tokens`, `messages` とllm-jp専用の `chat_template_kwargs` に現れます。`target_model_family` は出力せず、familyは `qa_id` の末尾で識別します。`target_tokenizer` とrevisionはresume情報へ分離します。Formatterはthinking本文を書き換えません。

トップレベルの `question`・`answer`・`thinking` と `messages` 内の同じ内容は、両方を保存する仕様です。この重複は旧形式が残っていることを意味しません。

- Qwen: `messages` のassistantに `reasoning_content: thinking`, `content: answer` を格納し、公式テンプレートが `<think>` wrapperを付けます。トップレベルの `reasoning_effort` を必ず学習時にテンプレート引数として渡してください。mediumには独自directiveを追加しません。
- llm-jp: `messages` は公式Tokenizerに渡せる２メッセージ形式です。userの `content` は質問の文字列、assistantは回答の `content` と推論本文の `thinking` を持ちます。systemやanalysis／finalの特殊トークンは公式テンプレートが生成します。
- llm-jpは `chat_template_kwargs` にeffortと固定日付を保持します。`template_messages` は保存しません。
- 両Datasetとも展開後の `text` は保存しません。messagesを正規のテンプレート入力として使い、question／answer／thinkingとの完全一致を検証します。生成時のテンプレート適用は一時的な整合性検証だけです。llm-jpのHarmony parser検証とモデル別thinking_tokens計算は維持します。

学習時のテンプレート適用例（生成した文字列は学習側で使用）:

```python
# Qwen: デフォルトのthinking/preserve_thinkingを使用
rendered = tokenizer.apply_chat_template(
    record["messages"], tokenize=False, add_generation_prompt=False,
    reasoning_effort=record["reasoning_effort"],
)
# LLM-jp: 固定日付も引き継ぐ
rendered = tokenizer.apply_chat_template(
    record["messages"], tokenize=False, add_generation_prompt=False,
    **record["chat_template_kwargs"],
)
```

共通生成のcanonicalラベルは既存どおりlow／medium／highです。Formatterの明示的な対応表はQwenでlow→low、medium→medium、high→xhigh、llm-jpで同名ラベルを維持します。これは以前からの生成ラベル対応であり、入力JSONLの誤ったhighを補正する規則ではありません。Qwen出力・移行時のreasoning_effortがhighならエラーにし、黙って変換しません。


### Cache / Resume / Failure

形式と長さに合格したcanonicalだけをJSONL journalへfsync付きで保存します。cache keyはsource ID・QA内容・metadata・effort・生成モデル・プロンプト本文/version・検証条件を含み、対象モデル設定は含みません。同じcanonical cacheで出力対象を追加しても再推論しません。Formatterの追加は新規クラスとprofile登録で対応できます。

同じCLI・設定で再実行すると、成功済みcanonicalを再検証して再利用します。対象ごとのJSONLの `qa_id`（旧形式はcanonical ID）を成功statusとして扱い、未出力targetだけを再処理します。新形式のTokenizer情報は `<出力パス>.resume.jsonl` に分離して保存し、このファイルも排他制御します。再開時には出力JSONLとresumeファイルをセットで保持してください。resumeファイルが欠損した新形式の行は、設定を推測してスキップせずfailureに記録します。例えばQwen成功・llm-jp失敗なら、次回の生成呼び出しは0回でllm-jpのみ再試行します。Tokenizerを変更する場合は新しい出力パスを指定してください。

末尾の途中書込みは再開時に復旧します。中間行の破損は黙って無視せず停止します。出力パスの重複・入力との衝突を拒否し、ファイルロックで同一出力への並行実行を防止します。canonical cacheが後から破損した場合はそのレコードを失敗扱いにし、自動再生成しません。修復する場合は該当cacheと出力を確認してください。

1件の生成・検証・Formatter・書込エラーで他レコードを停止しません。`failures_path` にsource ID、canonical ID、effort、`failed_step`, `format_errors`, `error` を記録します。全レコード処理後にsummaryを表示し、その実行で最終失敗が1件以上あればCLIは終了コード1を返します。過去のfailure履歴は削除されず、後から成功したレコードの履歴も残ります。履歴の行数だけで現在の未解決件数や終了コードは決まりません。`generator_calls` は生成メソッド呼出数、`llm_inference_calls` は通信再試行を含むHTTP送信試行数（接続待ちで失敗した試行も含み、サーバーで実際に推論した回数とは限りません）、`output_records` は再開前を含む出力総数です。

入力はHF streamingまたはローカルJSONLを有界windowで読み、Queue / worker pool、`max_in_flight`, `pipeline_batch_size`, httpx connection poolで処理します。生成APIはthinkingのみ要求し、返却contentを使用します。provider内部の非公開reasoningフィールドをfallbackとして保存しません。

### 検証とサンプル

関連するテストは次のコマンドで実行できます。

```bash
uv run python -m unittest \
  test.test_reasoning_effort_dataset \
  test.test_reasoning_effort_source_context \
  test.test_reasoning_effort_output \
  test.test_reasoning_effort_progress -v
# ダウンロード済み公式Tokenizerでの追加検証:
REASONING_REAL_TOKENIZERS=/path/to/hf/cache \
  uv run python -m unittest test.test_reasoning_effort_dataset -v
```

現行の出力形式は [構造化サンプルと説明](examples/reasoning_effort_dataset/structured_samples/REPORT.md) を参照してください。各モデル1件の固定fixtureで、実LLM生成ではありません。

[旧Mockサンプル](examples/reasoning_effort_dataset/sample_output/)は、初期実装時の旧出力形式を残した資料です。3回のMock HTTP呼出しから両モデル各3件を作成した記録であり、現在の保存スキーマの見本には使わないでください。[thinking表示](examples/reasoning_effort_dataset/sample_output/thinking_samples.md) は本文の例として参照できます。token範囲は全effort 1〜4096へ広げた構造検証用です。

`run_mock_sample.py` は現在の軽量出力から除去済みの `canonical_record_id`・`canonical_reasoning_effort` を出力レコードの比較で参照するため、新規出力での実行時に `KeyError` になる箇所が残っています。現行形式の検証には上記テストと構造化サンプルを使用してください。

### 仕様確認元・制約

2026-09-11に以下の公式情報と公開Datasetの先頭レコードを確認しました。

- [Qwen3.8公式chat template](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/chat_template.jinja): reasoning_contentとeffortの条件付け。
- [llm-jp公式chat template](https://huggingface.co/llm-jp/llm-jp-4-8b-thinking/blob/main/chat_template.jinja)、[Cookbook](https://github.com/llm-jp/llm-jp-4-cookbook): thinking/contentからHarmonyへの変換とカスタムTokenizer。
- [llm-jp公開SFT Dataset](https://huggingface.co/datasets/llm-jp/llm-jp-4-thinking-sft-data): channelとcontent blockを持つmessages。
- [入力validated QA](https://huggingface.co/datasets/ikedachin/imabari_wiki_qa_v4_validated): question/thinking/answer/messagesとMarkdown構造。
- [Transformers chat template](https://huggingface.co/docs/transformers/chat_templating): 学習時は `add_generation_prompt=False`。

構造Validatorは、自然な今治弁・事実の正しさ・意味的な水増し・見出しのない最終回答混入まで保証するものではありません。これらはプロンプトで制約し、実生成後の品質確認が必要です。contextを設定した場合は生成根拠として使い、出力のuser messageは元questionを維持します。ファイルロックはmacOS/Linuxの `fcntl` を使用します。`uv.lock` はGit管理対象外です。依存関係は `pyproject.toml` または `requirements.txt` に従って環境に導入してください。

### 元記事全文をQAのidで照合する設定

Reasoning Effort生成では、以下の設定により `test_output/cpt/wiki/all.jsonl` の全文を根拠としてプロンプトへ渡します。実行設定・設定テンプレートで有効化しています。

```yaml
source_context:
  enabled: true
  path: ./test_output/cpt/wiki/all.jsonl
  qa_id_field: id
  source_id_field: id
  text_field: text
  chunk_index_field: chunk_index
  mode: concatenate_all
```

QA自身の識別子は引き続き `fields.id: qa_id`、記事との照合は `source_context.qa_id_field: id` です。元記事は起動時に一度読み込み、同じidの全chunkを `chunk_index` 昇順で並べ、空行で結合します。同じchunk番号の行はファイル内の順序で全件保持し、重複の警告とsummaryの件数を記録します。整数idと文字列idは文字列表現で照合します。

QAごとにeffort展開前に照合し、結合した全文を `generation_context` として3 effortのプロンプトへ共通で渡します。有効時はこの全文を優先し、無効時は従来の `fields.context` を使います。全文の要約・切捨ては行いません。APIの入力上限超過は生成失敗として記録し、他QAの処理を継続します。question、answer、出力messagesの仕様は維持します。

容量を抑えるため、`generation_context` は新規のDataset・生成cacheには保存しません。元記事全文は生成プロンプトとcache keyの計算に使用し、記事id・本文ハッシュ等の `source_context_metadata` は内部生成cacheに保存します。旧cacheからDatasetを新規出力する場合も本文は除外します。すでに保存済みのファイルは自動で書き換えないため、既存行の容量は変わりません。

内部生成cacheの `source_context_metadata` に記事id、参照ファイルのパス、全chunk番号、結合本文のSHA-256を保存します。最終Datasetには保存しません。全文とmetadataをcache keyへ反映するため、contextを使わずに生成した旧レコードや、本文・参照パスが異なるレコードとはキーが変わります。保存済みcacheから `generation_context` だけを除去してもキーを維持すれば、この除去自体は再利用を妨げません。通常設定ではcache・両Dataset・failureのファイル名を `reasoning_effort_with_context` 系へ分離しており、旧成果物を保持します。同じ本文・設定での再開は推論0回です。

id不一致、QAのid欠損、記事の本文欠損・不正なchunk番号は、QA単位の `source_context_resolution` failureとし、effort展開も推論も行いません。記事の一部だけが欠損している場合も、残りのchunkだけで生成することはありません。参照ファイルの欠損、不正JSON、照合不能な元データ行のid欠損は生成開始前に停止します。参照元と出力のパス衝突も拒否します。

追加の結合・cache・失敗処理テスト:

```bash
python -m unittest test.test_reasoning_effort_source_context -v
```

既存の手書きMockサンプルは記事idを持たない独立fixtureなので、サンプルスクリプトでは `source_context.enabled: false` を明示しています。

### 生成プロンプトと検証失敗の診断

出力形式をsystemメッセージにも指定し、長いcontextの見出しや文章を出力の形式として引き継がないようにしています。提出するMarkdownの最初の文字は `#`、先頭の空白・空行は0文字と明示します。low / medium / highのプロンプトには空行込みの具体例と提出前の確認を追加しています。

プロンプト本文とsystemの出力指示はcache keyに含まれます。指示を変更した場合、旧指示による生成結果を流用しません。検証条件の緩和や、先頭空行の自動削除は行っていません。

実行中は `Thinking request`（QA ID・effort・試行番号）、`Thinking rejected`（検証段階・エラーコード）、`Thinking validated`（確定effort・token数）を逐次表示します。再試行上限に達した不合格文章はfailure JSONLの `rejected_thinking` に保存します。この文章は成功cacheやSFT Datasetには入りません。プロンプト・コード変更は、動作中のプロセスを停止して再実行すると反映されます。

生成用Qwen3.8では、内部thinking有効時の実応答に先頭改行2文字が残るケースを確認しました。指示を強めても残ったため、既定設定は `generator.generation.chat_template_kwargs.enable_thinking: false` と `generator.use_reasoning_effort: false` にしています。生成モデル内部のthinkingを無効にし、提出用の今治弁Markdownをcontentへ直接生成します。Canonical low / medium / highの違いは各プロンプトで指示し、Dataset側のeffort mapping・thinking領域は維持します。これは学習用Qwenテンプレートのthinking処理とは別設定です。現行FormatterはQwenへ `reasoning_effort` だけを渡し、thinking関連フラグは固定revisionのテンプレート既定値を使用します。

長さの再試行では、エラーコードに加えてreference tokenizerによる前回の実測token数と許容範囲を返します。前回の生成本文は再送せず、元のプロンプトに最新の修正指示を付けて生成し直します。モデルが不足・超過の程度を判断できるようにします。検証失敗を無条件で成功扱いにすることはありません。

### 通信失敗と並列数の確認

`max_in_flight` はeffortジョブのworker数、`max_connections` はHTTP接続上限です。workerが接続上限を超えて動くと、余ったジョブは接続プールを待ちます。`pool_timeout` はこの待ち時間、`connect_timeout` は接続確立、`read_timeout` は応答の読み取り待ちに使われます。`read_timeout` を増やしても接続プール待ちの上限は変わりません。

接続待ちが疑われる場合は、まずworker数を接続上限以下に下げて比較します。通信設定だけの変更はcanonical cache keyに含まれません。共有テンプレートは `max_in_flight: 8`・`max_connections: 16` ですが、ローカルYAMLの値は別途確認してください。

`failed_step: generation` は生成処理中の例外、`thinking_format_validation`・`thinking_length_validation` は生成本文の検証失敗です。現状は例外の `str(exc)` のみを保存しており、通信例外によっては `error` が空文字になります。例外型や各通信試行の所要時間は記録しないため、空文字だけから `PoolTimeout` と断定できません。

`max_retries: 3` は生成メソッド1回につき初回と追加3回までの送信試行を許します。対象HTTPステータスは408・409・425・429・500・502・503・504です。通信例外、空content、出力打切り等も再試行しますが、同じpayloadを使用し、自動で同時実行数やtoken上限は調整しません。形式・長さ検証の再試行はこれとは別枠です。

### ローカル実行設定とGit管理

`yamls/create_reasoning_effort_dataset.yaml` は実サーバーやローカル出力先を設定するためGit管理対象外です。共有するのは `*_settings_format.yaml` の設定テンプレートです。新しくcloneした環境では、初回に次のコマンドで実行設定を作成し、接続先を編集してください。既存の実行設定は上書きしません。

```bash
cp -n yamls/create_reasoning_effort_dataset_settings_format.yaml \
  yamls/create_reasoning_effort_dataset.yaml
```

テストも共有テンプレートから設定を読み、ローカル実行設定には依存しません。`.env.*` は除外しますが、値を伏せた `.env.example` と `.env.sample` は共有できます。MockサンプルのJSONL・Markdownは共有し、実行時のロック・失敗ログは除外します。


### Reasoning Effortの軽量出力と既存ファイルの移行

実行中はターミナルの最下部にQA単位の進捗バーを表示します。既存のrequest・validated・rejectedログは `tqdm.write` でバーの上へ表示します。ローカルJSONLは最初に空行を除く行数を数え、不正JSON行も失敗として処理するため総数に含めます。Hugging Faceのストリーミングでは総数・割合・残り時間を推測せず、完了件数・経過時間・速度を表示します。

１QAの全effortと対象出力の処理が終了した時点で完了数を増やします。再試行は完了数を増やさず、入力不備や最終失敗は処理完了に含めます。中断中のQAは完了扱いにしません。横の件数は「生成成功＝新たに検証を通過したcanonicalの処理件数」「cache再利用＝canonical再利用件数」「最終失敗＝入力・生成・対象出力での最終失敗件数」「実行中＝処理中のeffortジョブ数」です。単位が異なるため、これらを合計してもQA数にはなりません。summaryには `completed_qa` を追加します。

標準エラーがターミナルでない場合はバーを無効にし、開始・終了時と処理中30秒ごとに通常の進捗ログを出します。既存の実行ログは引き続き表示し、この機能のログからANSI色制御文字を除きます。通常のターミナルでは約１秒ごとに状況を更新します。残り時間は参考値です。

最終Datasetは次の許可リスト順で保存します。未知の元QA項目は追加されません。

```text
qa_id, source_qa_id, id, chunk_index, question, answer, thinking,
reasoning_effort, eval, messages,
chat_template_kwargs（llm-jpのみ）, thinking_tokens,
thinking_generator, source_metadata
```

内部canonicalレコードと生成cache、failureログはこの許可リストの対象外です。messagesは両モデルで保存し、chat_template_kwargsはllm-jpのみ保存します。template_messagesとtextは保存しません。旧形式の既存行は自動変更せず、再開後に追加する行だけ新形式になります。

#### 単一JSONLの移行

1つのモデル出力を新しいパスへ変換するには、生成処理を終了してから次を実行してください。モデル種別はqa_idから自動判別するため、Qwen・llm-jp共通です。このコマンドはキャッシュや失敗履歴を移行しません。

```bash
uv run python migrate_reasoning_effort_output.py \
  --file /path/old.jsonl \
  --output /path/compact.jsonl
```

元ファイルは変更しません。既存の出力パスへの上書き、不正JSON、混在モデル、重複ID、必要な情報の欠損は拒否します。全行の確認後に新しいJSONLと `.resume.jsonl` を作成します。新形式を再変換する場合は、入力と同じ場所の `.resume.jsonl` も必要です。移行コードだけは旧形式のtemplate_messagesを読み取り、旧Harmony形式messagesからの移行を支援します。新規生成はmessagesを直接作り、template_messagesを経由しません。移行でもtext・template_messages・Qwenのchat_template_kwargsを除外します。TokenizerやLLMを呼ばず、質問・回答・thinking等は変更しません。

変換先で生成を再開する場合は、YAMLの対象出力パスを変換先に変更し、従来の生成cacheパス・生成設定を維持してください。変換先の `.resume.jsonl` も一緒に保持します。Qwenとllm-jpはそれぞれ別ファイルへ変換してください。学習自体にはresumeファイルは不要です。


#### ディレクトリ全体の移行・統合

[migrate_reasoning_effort_directory.py](migrate_reasoning_effort_directory.py) は、次の固定ファイル名を扱う一時移行ツールです。YAMLは読み取りません。共有設定テンプレートの出力ファイル名とは異なるため、対象ディレクトリ内の名前を確認してください。

| ファイル | 移行内容 |
| --- | --- |
| `qwen38.jsonl`・`llmjp4.jsonl` | 軽量出力へ変換し、既存の新側データと統合 |
| 各出力の `.resume.jsonl` | canonical ID・Tokenizer名・revisionを保存 |
| `.generation_cache.jsonl` | キー・本文・内部metadataを保持し、トップレベルの `generation_context` を除去 |
| `failures.jsonl` | 失敗履歴を統合し、JSON値が完全一致する重複だけ除去 |

入力側は両モデルの出力とキャッシュが必須です。新形式の入力には隣接するresume情報も必要です。旧側と新側に同じcanonical IDがある場合は新側を優先します。統合後の出力とキャッシュについてQA ID・質問・回答・thinking・effortの一致を検査し、不一致や対応するキャッシュの欠落はエラーにします。入力JSONL内の出力ID重複も拒否します。

生成処理を停止し、リポジトリルートから実行します。次は検証のみです。

```bash
uv run python migrate_reasoning_effort_directory.py \
  --source ./test_output/reasoning_effort_with_context_old \
  --destination ./test_output/reasoning_effort_with_context
```

反映する場合は `--apply` を追加します。

```bash
uv run python migrate_reasoning_effort_directory.py \
  --source ./test_output/reasoning_effort_with_context_old \
  --destination ./test_output/reasoning_effort_with_context \
  --apply
```

上記2ディレクトリは既定値なので、パス引数は省略できます。`--apply` がない場合も出力ディレクトリ・ロックファイルと一時作業領域は作成しますが、対象JSONLを置換しません。入力・出力は別ディレクトリで、相互に包含しない必要があります。

反映時は新側の `migration-backup-日時-識別子/` に反映前のファイル、`manifest.json`、変換済み一式の `prepared/` を保存してから対象ファイルを置換します。旧側のデータ本文は変更しません。同じ入力で再実行しても出力IDや完全一致の失敗履歴は増えませんが、`--apply` ごとにバックアップを作成します。

置換はファイルごとです。捕捉できる置換エラーでは反映済みファイルを戻しますが、強制終了・電源断を含むディレクトリ全体の一括更新は保証しません。復元時は生成処理を止め、manifestの `previous_files` とバックアップ一式を照合してください。ロックファイルの存在だけは実行中を意味せず、OSのロック取得に失敗した場合に移行を停止します。

移行ではLLM・Tokenizerを呼ばず、token数の再計測や事実性・方言品質の再評価はしません。キャッシュキーも再計算しないため、生成設定・プロンプト・元記事が変わった場合のキャッシュヒットは保証しません。移行後はYAMLの両出力・cache・failureパスを移行先に合わせ、resume情報も保持して再開してください。成功済みQAの過去の失敗履歴が残るのは仕様です。

#### 進捗表示の切り替え


進捗表示は環境変数 `REASONING_PROGRESS=auto|bar|log` で選択できます（既定はauto）。autoは標準エラーの端末判定を使うため、標準出力だけを `tee` に渡してもバーを表示します。端末判定が得られない場合はbar、通常ログのみの場合はlogを指定してください。総件数不明の場合はbar指定でも割合は表示しません。

```bash
REASONING_PROGRESS=bar uv run python main_create_reasoning_effort_dataset.py \
  --settings_path ./yamls/create_reasoning_effort_dataset.yaml
```
