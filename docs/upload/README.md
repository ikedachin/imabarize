# アップロード：Hugging FaceへのJSONL・Parquet公開

[トップ](../../README.md) · [メイン処理](../main/README.md) · [特殊なデータセット](../special_datasets/README.md)

コマンドはリポジトリルートで実行します。先に[共通セットアップ](../../README.md#セットアップ)を行い、入力とアップロード先を自分の環境に合わせて変更してください。

## ツールの選び方

| ツール | 目的 | 入力 | アップロード形式 |
| --- | --- | --- | --- |
| `main_upload_dataset.py` | CPT / QA / GRPOの生成結果を公開 | `all.jsonl` を含むディレクトリ | JSONL（分割時はREADMEも対象） |
| `tool_upload_for_llmjp4.py` | LLM-jp 4用の型を明示して公開 | Reasoning Effortの `llmjp4.jsonl` | 固定スキーマのParquet |
| `tool_upload_parquet_to_huggingface.ipynb` | 任意のデータをParquet化して公開 | JSONL / JSON配列 / CSV / Parquet | 型を自動推定したParquet |
| `tool_upload_imabari_qa_v4_to_huggingface.ipynb` | 各行にUUIDv4の `qa_id` を付けて公開 | Q&AのJSONL | 9:1分割のJSONL |

Reasoning EffortのLLM-jp出力は、`messages` の不均一な構造を扱うため、専用CLIを使うと固定スキーマで検証できます。どのツールもQ&Aやthinkingの生成、標準語化は行いません。

## 1. 通常のCPT / QA / GRPO：`main_upload_dataset.py`

### 入力と前処理

`--dataset_path` でディレクトリを指定します。省略時は `--settings_path` のYAMLから `output_path` を読みます。アップロード用の[設定テンプレート](../../yamls/upload_dataset_settings_format.yaml)に加え、生成時のYAMLもそのまま使えます。Reasoning Effortの `output.*.path` 構造には対応しません。

- 通常は `all.jsonl` が必須です。`stats.json` や失敗ログ・cacheは対象外です。
- `--include-splits` を付けると既存の `train.jsonl` / `validation.jsonl` も対象になります。
- `--split-on-upload` は `all.jsonl` を前処理してシャッフルし、train / validationを作ります。このモードでは生成した2ファイルと、入力ディレクトリに存在すれば `README.md` をアップロードし、`--include-splits` は無視します。残るレコードが2件以上必要です。
- `--dataset-kind cpt` は `copyright_mitigation` が真の行だけを残します。既定の `auto` も、そのキーがある行には同じ条件を適用します。`qa` / `grpo` は全行を残します。
- 既定で `item_id` を除外します。追加削除するキーは `--exclude-upload-key` を繰り返すか、`--exclude-upload-keys key1,key2` で指定します。行の選別後にキーを削除します。

前処理と分割は一時ディレクトリで行い、元JSONLは変更しません。

### 確認とアップロード

まず対象と件数を確認します。`--dry-run` でも `--repo_id` は必要ですが、トークンは不要です。

```bash
uv run python main_upload_dataset.py \
  --repo_id YOUR_NAME/YOUR_DATASET \
  --dataset_path ./test_output/imabari_qa \
  --dataset-kind qa \
  --split-on-upload \
  --validation-ratio 0.1 \
  --split-seed 42 \
  --dry-run
```

書き込み権限のあるトークンを `HF_TOKEN` 環境変数に設定し、同じ条件でアップロードします。このCLIは `.env` や保存済みログインを自動利用しないため、`--hf_token` への明示的な指定が必要です。

```bash
uv run python main_upload_dataset.py \
  --repo_id YOUR_NAME/YOUR_DATASET \
  --hf_token "$HF_TOKEN" \
  --dataset_path ./test_output/imabari_qa \
  --dataset-kind qa \
  --split-on-upload \
  --validation-ratio 0.1 \
  --split-seed 42 \
  --private
```

CPTの既存splitを含める例です。`copyright_mitigation: false` の行は除外されます。

```bash
uv run python main_upload_dataset.py \
  --repo_id YOUR_NAME/YOUR_CPT_DATASET \
  --settings_path ./yamls/create_cpt_dataset_settings_format.yaml \
  --dataset-kind cpt \
  --include-splits \
  --exclude-upload-key source_file \
  --dry-run
```

## 2. LLM-jp 4専用：`tool_upload_for_llmjp4.py`

Reasoning Effortの今治弁版・標準語版を読み込み、`messages` を `List(Json)`、会話日付をtimestampとして保存します。train / validationのParquetを作成して再読込し、全値とスキーマを検証します。

ファイル名は **`tool_upload_for_llmjp4.py`** です。スクリプト内の説明に残っている旧名 `tool_upload_standard_japanese_llmjp4.py` は使用しません。

### 入力の条件

- `qa_id` は空でない一意な文字列。`question` / `answer` / `thinking` / `thinking_generator` も空でない文字列。
- `messages` はuserの質問と、assistantの `content`（回答）・`thinking`（思考）からなる2件構造で、トップレベルの値と一致すること。
- `reasoning_effort` は `low` / `medium` / `high`、`chat_template_kwargs.reasoning_effort` と一致すること。
- `thinking_tokens`、`chat_template_kwargs.conversation_start_date`、`source_metadata` 等の所定の項目が必要です。未知の項目や矛盾するIDがある場合は停止します。
- 既定では `eval: null` の未評価データを想定します。評価済みデータは `--allow-evaluated` で文字列の `eval` も許可できます。
- 分割には2件以上必要です。`--validation-ratio` の既定は `0.1`、`--seed` の既定は `42` です。

固定スキーマには `List(Json)` 対応の `datasets` が必要です。非対応の環境では、スクリプトに記載された依存バージョンの例を指定して実行できます。

```bash
uv run --with datasets==5.0.1 python tool_upload_for_llmjp4.py \
  --input ./test_output/create_qa_v5_difficult_reasoning_effort/llmjp4.jsonl \
  --output-dir ./test_output/create_qa_v5_difficult_reasoning_effort/upload_parquet_llmjp4 \
  --repo-id YOUR_NAME/YOUR_LLMJP4_DATASET \
  --allow-evaluated \
  --dry-run
```

`--dry-run` でもローカルParquetを生成・検証し、同名の出力を上書きします。アップロードは行いません。出力は次の2ファイルです。

```text
<output-dir>/data/train-00000-of-00001.parquet
<output-dir>/data/validation-00000-of-00001.parquet
```

確認後、同じコマンドの `--dry-run` を外してアップロードします。`--private` を追加すると新規リポジトリを非公開で作れます。認証はルートの `.env` の `HF_TOKEN`、設定済みの環境変数、保存済みのHugging Faceログインに対応します（既存環境変数を優先）。

省略時は `test_output/temporary_standard_japanese/llmjp4.jsonl` とスクリプト内の既定アップロード先を使います。別データでの利用時は `--input` / `--output-dir` / `--repo-id` を明示してください。

## 3. 汎用Parquet Notebook

[`tool_upload_parquet_to_huggingface.ipynb`](../../tool_upload_parquet_to_huggingface.ipynb) を開き、プロジェクトのPython環境をカーネルに選びます。

1. 設定セルの `PROJECT_DIR`、`INPUT_FILES`、`OUTPUT_DIR`、`REPO_ID`、`PRIVATE` を編集します。
2. `SHUFFLE`、`SEED`、`VALID_RATIO`、`VALID_SPLIT_NAME` を設定します。`VALID_RATIO = 0` ならtrainのみです。
3. 読込・分割・Parquet作成のセルを順に実行し、表示されるスキーマと件数を確認します。同じ形式・カラムの入力ファイルを複数指定できます。
4. 最後のアップロードセルを実行すると、今回生成したParquetをHubへ送信します。ローカル生成だけならその前で止めます。

カラム変換、項目削除、ID変更、READMEの生成はしません。認証は `PROJECT_DIR/.env` の `HF_TOKEN`、環境変数、保存済みログインを利用します。

## 4. Q&AへUUIDを付けるNotebook

[`tool_upload_imabari_qa_v4_to_huggingface.ipynb`](../../tool_upload_imabari_qa_v4_to_huggingface.ipynb) は、全行を保持して一意なUUIDv4の `qa_id` を各行へ付与し、通常CLIを呼び出して9:1に分割・公開します。`id + chunk_index` や内容が重複していても保持します。

1. 設定セルの `SOURCE_FILE`、`UPLOAD_DIR`、`REPO_ID`、`PRIVATE`、`VALIDATION_RATIO`、`SPLIT_SEED` を編集します。Notebook名はv4ですが、現在の入力既定値は `test_output/temporary_standard_japanese/llmjp4.jsonl` です。
2. JSONL作成セル、Dry runセルで、出力と件数を確認します。
3. アップロードセルを実行します。認証はセル内で `.env` / `HF_TOKEN` を読み、未設定なら入力を求めます。

## 公開時の分割とファイル更新

各ツールの分割はレコード単位です。同じ質問のeffort違いや同一記事のチャンクがtrain / validationにまたがる可能性があります。質問・記事単位の分割が必要なら、先にその単位で入力を分けてください。

`--private` / `PRIVATE` は新規作成時の公開設定です。既存リポジトリの公開・非公開は変更しません。同名のアップロード先ファイルは更新し、対象外の既存ファイルは削除しません。分割構成を変更する場合は、Hub側の旧splitファイルやデータセットREADMEの設定も合わせて確認してください。
