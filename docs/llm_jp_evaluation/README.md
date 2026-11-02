# モデルの回答生成と別モデルによる評価

このプログラムは、評価用データセットの問題を**評価対象モデル**に解かせ、その回答を**別の Judge モデル**に採点させます。対象モデルには問題文だけを送り、基準回答は渡しません。Judge モデルには問題文・基準回答・対象モデルの回答を渡します。

評価対象として次のモデル ID が設定されています。

[ikedachin/llm-jp-4-8b-thinking-imabari-chikamiyama-v14](https://huggingface.co/ikedachin/llm-jp-4-8b-thinking-imabari-chikamiyama-v14)

```text
評価用データセット
  ├─ 問題文 ──→ 評価対象モデル ──→ 生成回答を保存
  └─ 基準回答 ──────────────────┐
                                ↓
              問題文・基準回答・生成回答 ──→ 別の Judge モデル
                                                 ↓
                                    スコア・判定・理由を保存
```

回答生成と採点は別々に実行できます。一度に両方を行うモードもあり、その場合は各問題の回答ができると、その問題の採点を開始します。複数の問題は worker-pool により並行して処理されます。

各問題は、今治弁と標準語のそれぞれについて `reasoning_effort` の `low`・`medium`・`high` を試します。したがって標準設定では、問題数の **6 倍**の回答生成と採点を行います。

| 条件 | 指定する `valid_channels` | `reasoning_effort` |
| --- | --- | --- |
| 今治弁 | `analysis_imabari`, `final_imabari`, `commentary` | `low`, `medium`, `high` |
| 標準語 | `analysis`, `final`, `commentary` | `low`, `medium`, `high` |

提供された v14 のチャットテンプレートでは、`valid_channels` はシステムメッセージに利用可能なチャネルを宣言し、`reasoning_effort` は推論強度を指定します。生成開始位置は `<|start|>assistant` で、チャネルはテンプレートによって固定されません。チャネルの宣言や推論強度は、実際の生成言語・生成チャネル・推論量を保証するものではありません。両スタイルのチャネル名と effort の組み合わせは設定で変更できます。

## 必要なもの

- Python 3.11 以上と、このリポジトリの依存関係。セットアップはリポジトリルートで `uv sync` を実行します。
- 評価用の質問と基準回答を含む JSON / JSONL ファイル。
- 対象モデルを読み込んだ OpenAI 互換 `/v1/chat/completions` サーバー。
- 別の Judge モデルを利用できる OpenAI 互換 API。設定例では OpenRouter を使います。

対象モデルのダウンロードや推論サーバーの起動は、この評価プログラムには含まれていません。サーバーでは対象モデルの tokenizer・chat template を使用し、最終回答を API の `message.content` に返すようにしてください。モデルへのアクセス権、必要なメモリ、サーバー固有のオプションは実行環境に合わせて確認してください。

例えば vLLM を別途導入した環境では、サーバー起動コマンドは次の形式です。サーバーのバージョンやモデルに合うオプションを適宜追加してください。

```bash
vllm serve ikedachin/llm-jp-4-8b-thinking-imabari-chikamiyama-v14
```

## 1. 評価データを用意する

JSONL の例です。1 行が 1 問に対応します。

```jsonl
{"id":"qa-1","question":"今治市は何県にありますか？","answer":"愛媛県です。"}
{"id":"qa-2","question":"問題文の例です。","answer":"基準回答の例です。"}
```

| 項目 | 意味 |
| --- | --- |
| `id` | 問題を識別する ID。数値の場合も文字列に変換されます。 |
| `question` | 対象モデルに送る問題文。空でない文字列が必要です。 |
| `answer` | Judge が比較に使う基準回答。空でない文字列が必要です。 |
| `chunk_index` | 任意の区分番号。`id` と組み合わせて問題を識別します。 |

`id` と `chunk_index` の組み合わせが重複すると処理を停止します。`chunk_index` がない行は、同じ `id` を重複して使えません。JSONL のほか、JSON オブジェクトと JSON 配列も利用できます。ディレクトリを指定すると、配下の `.json` と `.jsonl` を再帰的に読み込みます。

入力キーが `qa_id` の場合などは、次の手順の YAML の `fields` で指定できます。入力ファイルに含まれる `thinking` などの追加項目は、対象モデルには送られません。

## 2. 設定する

以降のコマンドはリポジトリルートから実行します。YAML 内の相対パスも、YAML ファイルの位置ではなく、実行時のカレントディレクトリを基準にします。

```bash
cp -n yamls/evaluate_llm_jp_settings_format.yaml \
  yamls/evaluate_llm_jp_settings.yaml
```

コピーした `yamls/evaluate_llm_jp_settings.yaml` を編集します。`cp -n` は既存の設定ファイルを上書きしません。

```yaml
eval_qa_path: ./test_output/eval_qa/validation.jsonl
output_path: ./test_output/llm_jp_chikamiyama_v14_eval
judge_prompt: ./prompts/judge_eval_qa/judge_answer.md
style_judge_prompt: ./prompts/judge_eval_qa/judge_llm_jp_style.md
fields:
  id: id                  # データの ID が qa_id の場合は qa_id に変更
  question: question
  answer: answer

workers: 16
reasoning_efforts: [low, medium, high]
styles:
  imabari:
    label: 今治弁
    valid_channels: [analysis_imabari, final_imabari, commentary]
  standard:
    label: 標準語
    valid_channels: [analysis, final, commentary]

target:
  base_url: http://localhost:8000/v1
  model: ikedachin/llm-jp-4-8b-thinking-imabari-chikamiyama-v14
  max_in_flight: 8
  timeout: 600
  max_attempts: 4
  retry_wait_seconds: 0.5
  generation:
    max_tokens: 4096
    temperature: 0
    chat_template_kwargs:
      conversation_start_date: "2026-09-11"

judge:
  base_url: https://openrouter.ai/api/v1
  model: qwen/qwen3.5-27b
  api_key_env: OPENROUTER_API_KEY
  max_in_flight: 8
  timeout: 180
  max_attempts: 4
  retry_wait_seconds: 0.5
  generation:
    max_tokens: 2048
    temperature: 0
    top_p: 1.0
```

`target` は回答生成用、`judge` は採点用です。`model` は各サーバーが受け付けるモデル名に合わせます。Judge をローカルサーバーなどに変更する場合も、`judge.base_url` と `judge.model` を設定してください。プログラムは対象モデルと Judge が別モデルであることを強制しないため、設定時に確認してください。

API キーが必要な場合は `api_key_env` に**環境変数名**を書き、キーの値は環境変数に設定します。OpenRouter を使う設定例では次のようにします。

```bash
export OPENROUTER_API_KEY='ご自身のAPIキー'
```

対象サーバーにもキーが必要なら、例えば `target.api_key_env: TARGET_API_KEY` を追加し、その環境変数を設定します。`api_key_env` を省略した場合はダミーの Bearer トークンを送ります。

`generation` 内の項目は各 API のリクエストに渡されます。対象モデルへの各リクエストでは `valid_channels` と `reasoning_effort` を `chat_template_kwargs` に追加します。設定例の `conversation_start_date` はテンプレート内の日付を固定して条件間・再実行間の比較をしやすくします。

**対象サーバーが `chat_template_kwargs` を実際に対象モデルのテンプレートへ渡すことを確認してください。** API が未対応で設定を無視すると、６条件に分けて保存されても、同じテンプレート条件で生成される可能性があります。サーバーのログやテンプレート適用結果で、`# Valid channels:` と `Reasoning:` が各リクエストに応じて変わることを確認してください。テンプレートにない `generate_thinking` / `thinking_channel` による開始チャネル指定は行いません。

## 3. 回答を生成する

まず、対象モデルで全条件の回答を生成します。この段階では Judge API に接続せず、Judge の API キーも必要ありません。

```bash
uv run python main_evaluate_llm_jp.py \
  -p ./yamls/evaluate_llm_jp_settings.yaml \
  -q ./test_output/eval_qa/validation.jsonl \
  --stage generate
```

回答は `output_path` の `answers.jsonl` に保存されます。途中で失敗・中断した場合は同じコマンドで、生成済みの条件をスキップして再開できます。`stats.json` の `remaining_generation_records` が `0` なら全条件の回答生成が完了です。

## 4. 保存済み回答を採点する

回答生成後に、同じ設定ファイルと評価データを指定して Judge を実行します。この段階では対象モデルに接続せず、対象モデルの API キーも必要ありません。Judge API のキーが必要なら、ここで環境変数に設定します。

```bash
export OPENROUTER_API_KEY='ご自身のAPIキー'

uv run python main_evaluate_llm_jp.py \
  -p ./yamls/evaluate_llm_jp_settings.yaml \
  -q ./test_output/eval_qa/validation.jsonl \
  --stage judge
```

保存済みの回答だけを採点し、生成されていない条件の回答を自動生成することはありません。一部しか生成されていない場合は利用可能な回答を採点し、未採点の条件が残るため終了コード `1` を返します。その後、生成・採点それぞれを再実行できます。保存済み回答が一つもない場合はエラーになります。

`--stage both` または `--stage` の省略時は、従来どおり回答生成と採点を続けて行います。

| 引数 | 意味 |
| --- | --- |
| `-p`, `--settings` | 設定 YAML。省略時は `yamls/evaluate_llm_jp_settings_format.yaml`。 |
| `-q`, `--eval-qa` | 評価データのファイルまたはディレクトリ。省略時は YAML の `eval_qa_path`。 |
| `-o`, `--output` | 出力先ディレクトリ。省略時は YAML の `output_path`。 |
| `--stage` | `generate`（回答生成）、`judge`（採点）、`both`（両方）。省略時は `both`。 |

実行中は、各 worker が処理した ID と判定ラベル、または失敗した段階を表示します。生成モードでは未生成の条件が、採点・両方のモードでは未採点の条件が残った場合、終了コードは `1` になります。

## 5. 結果を確認する

結果は `output_path` に保存されます。複数の問題が並行して完了するため、JSONL の出力順序は入力順とは限りません。`id` と `chunk_index` で対応を確認してください。

| ファイル | 内容 |
| --- | --- |
| `answers.jsonl` | 対象モデルの生成回答。ID、スタイル、effort、指定チャネル、取得できた最終チャネルなど。 |
| `all.jsonl` | 問題文、基準回答、生成回答、正確さとスタイルについての Judge の判定結果など。 |
| `failures.jsonl` | 生成または採点の失敗記録。失敗がなければ作成されないことがあります。 |
| `stats.json` | 採点件数、ラベル別件数、平均スコア、今回の失敗件数など。 |
| `manifest.json` | 同じデータと設定で再開するための識別情報。 |

`all.jsonl` の各行には、例えば次の項目があります。`key` はプログラムが再開時の識別に使う値です。

```json
{"id":"qa-1","chunk_index":null,"question":"今治市は何県にありますか？","reference_answer":"愛媛県です。","candidate_answer":"愛媛県にあります。","style":"standard","style_label":"標準語","reasoning_effort":"medium","valid_channels":["analysis","final","commentary"],"observed_final_channel":null,"judge_score":5,"judge_label":"correct","judge_reason":"基準回答と同じ内容です。","style_judge_label":"matches","style_judge_reason":"標準語の表現です。","target_model":"評価対象モデル名","judge_model":"Judgeモデル名","key":"識別用ハッシュ"}
```

標準の [Judge プロンプト](../../prompts/judge_eval_qa/judge_answer.md) は、表現が異なっても意味が一致すれば正解とし、今治弁や口語表現も不利にしないよう指示します。

| 項目 | 意味 |
| --- | --- |
| `judge_score` | 1〜5 の整数。5 は完全に正しく、1 は誤り・無関係など。 |
| `correct` | 基準回答と同じ内容を過不足なく答えている。 |
| `partially_correct` | 主要な内容は合っているが、不足や軽微な誤りがある。 |
| `incorrect` | 重要な内容が誤っている、または質問に答えていない。 |
| `unjudgeable` | 問題文や基準回答だけでは判定できない。 |

別の `style_judge_label` と `style_judge_reason` は、回答が指定された表現スタイルに合うかを示します。ラベルは `matches`、`partially_matches`、`does_not_match`、`unjudgeable` のいずれかです。短い回答など今治弁と標準語を区別できない場合は `unjudgeable` とするよう Judge に指示します。スタイル判定も LLM による推定であり、人手による確認が必要です。

`stats.json` の `generated_records`、`judged_records`、`labels`、`mean_score` は保存済みの全結果を対象にします。`mean_score` は `unjudgeable` を含む採点済み全件の平均で、採点結果がない場合は `null` です。`generated_this_run`、`judged_this_run`、`generation_failures`、`judge_failures` は今回の実行分です。失敗ログは追記されるため、後の実行で成功しても過去の失敗記録は残ります。

`input_questions` は入力問題数、`total_records` は問題数と評価条件を掛け合わせた件数です。`by_style_and_effort` には条件別の生成・採点件数、正確さとスタイルのラベル件数、平均点を保存します。`observed_final_channels` は回答生成時に観測できた最終チャネルの件数です。

Judge は LLM であり、その判定は基準回答・Judge モデル・プロンプトの品質に依存します。プログラムはスコア、ラベル、理由の形式を確認しますが、判定の正しさやスコアとラベルの整合性は保証しません。

## 並列数と再試行

`workers` は同時に処理を進める問題数の上限です。各問題は回答生成の後に採点されます。`target.max_in_flight` は回答生成 API、`judge.max_in_flight` は Judge API の同時リクエスト数をそれぞれ制限します。サーバー性能や API の利用制限に合わせて調整してください。キューは有界ですが、入力データ自体はメモリに読み込みます。

`max_attempts` は最初のリクエストを含む最大試行回数です。429 や一部の 5xx、タイムアウトなどは待ち時間を設けて再試行します。Judge が空でないものの採点タグの形式を誤った場合は、その項目を採点失敗として記録します。

対象モデルの返答に `<think>...</think>` や Harmony の `final` / `final_imabari` チャネルが含まれる場合は最終回答を抽出します。Harmony チャネルを取得できた場合は `observed_final_channel` に記録します。サーバーがチャネル情報を除去した通常の回答を返す場合は `null`、集計では `unknown` です。指定したスタイルと異なるチャネルが出力されても、実際の結果として記録します。

一部のサーバーでは Harmony の特殊トークンを除去し、`analysis_imabari ... assistant final_imabari ...` のようにチャネル名だけを返します。回答が `analysis` / `analysis_imabari` で始まる場合、この形式からも最終回答を抽出します。特殊トークンが除去された表現は通常の文章と完全には区別できないため、抽出結果を確認してください。

`finish_reason: length`、未完了の thinking、最終回答がない場合は失敗として扱います。サーバーが thinking と回答を区別せず通常のプレーンテキストとして返す場合、プログラムは両者を区別できません。接続先の出力形式を確認してください。

## 再開と再評価

同じ入力と設定で再実行すると、採点済みの「問題 × スタイル × effort」はスキップします。回答生成に成功して採点に失敗した条件は、`answers.jsonl` の回答を使って採点を再開します。生成に失敗した条件は、回答生成からやり直します。

入力データや対象モデルの設定を変更した場合、結果の混在を防ぐため既存の出力先では再開できません。Judge の設定とプロンプトは採点開始時に固定され、それ以降に変更した場合も別の出力先が必要です。対象モデルと Judge の設定には並列数やタイムアウトなども含まれます。生成と採点の間に Judge を選ぶことは可能です。以前の一体実行で作成された出力は、従来どおり Judge の設定を含めて照合します。全件を新たに採点したい場合にも別の出力先を使えます。

### 旧形式の生成結果を修復する

以前のバージョンで `analysis_imabari ... assistant final_imabari ...` を回答全体として保存していた場合は、**生成プログラムを停止してから**、新しい出力先に最終回答を抽出できます。元ファイルは変更されません。採点結果が存在する出力は、このコマンドでは修復できません。

```bash
uv run python tool_recover_llm_jp_answers.py \
  --source ./test_output/llm_jp_chikamiyama_v14_eval \
  --output ./test_output/llm_jp_chikamiyama_v14_eval_recovered
```

抽出できなかった行は新しい出力先の `rejected_answers.jsonl` に保存し、未生成扱いにします。未生成の条件は以下のコマンドで生成し直せます。

```bash
uv run python main_evaluate_llm_jp.py \
  -p ./yamls/evaluate_llm_jp_settings.yaml \
  -q ./test_output/eval_qa/all.jsonl \
  -o ./test_output/llm_jp_chikamiyama_v14_eval_recovered \
  --stage generate
```


採点時も同じ `-o` を指定し、`--stage judge` で実行します。新しい出力先は作成済みでないものを指定してください。
```
uv run python main_evaluate_llm_jp.py \
  -p ./yamls/evaluate_llm_jp_settings.yaml \
  -q ./test_output/eval_qa/all.jsonl \
  --stage judge
```

## 関連ファイル

- [実行プログラム](../../main_evaluate_llm_jp.py)
- [評価パイプライン](../../pipelines/evaluate_llm_jp.py)
- [設定テンプレート](../../yamls/evaluate_llm_jp_settings_format.yaml)
- [Judge プロンプト](../../prompts/judge_eval_qa/judge_answer.md)
- [表現スタイルの Judge プロンプト](../../prompts/judge_eval_qa/judge_llm_jp_style.md)
- [テスト](../../test/test_evaluate_llm_jp.py)
- [既存の回答のみを採点するプログラム](../../main_judge_eval_qa.py)
