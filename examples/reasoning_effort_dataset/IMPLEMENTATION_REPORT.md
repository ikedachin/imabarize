# Reasoning Effort Dataset Generator 実装報告

## 現行仕様（2026-09-14 ドキュメント見直し）

この報告の「1–25」と2026-09-12の追記は、各作業時点の記録です。旧出力の保存項目、当時のテスト件数、実LLM未接続、Gitの状態は現在の状態を表すものではありません。現行の操作手順は [README](../../README.md)、出力例は [構造化サンプル](structured_samples/REPORT.md) を参照してください。

現行実装を読み直して確認した相違点:

- 最終Datasetは `output_schema.compact_output` の許可リストで保存する。Qwenは13項目、llm-jpは14項目。`source_metadata` は元QAの `qa_id`・`id`・`chunk_index` のみ。
- 両モデルの `messages` はuserとassistantの2要素。assistantは回答の `content` と、Qwenでは `reasoning_content`、llm-jpでは `thinking` を持つ。トップレベルとmessagesの本文は同じ内容を保存する。
- `text`・`template_messages` は保存しない。`chat_template_kwargs` はllm-jpだけに保存する。テンプレート展開とHarmony検証は生成時の一時的な処理。
- 対象Tokenizer名とrevisionは `<出力>.resume.jsonl` へ分離する。キャッシュにはcanonical ID、reference token数、内部metadataを保持するが、新規生成では `generation_context` を保存しない。本文は生成プロンプトとキー計算に使用する。
- 生成側は共有テンプレートで `enable_thinking: false`・`use_reasoning_effort: false`。effortは各プロンプトで制御し、出力ラベルはQwenのhigh→xhigh、llm-jpのhigh→highを維持する。
- 通常の実行YAMLはGit管理対象外。共有するのは `create_reasoning_effort_dataset_settings_format.yaml`。初回コピーと接続先の編集はREADMEを参照。
- `sample_output/` は旧形式の履歴資料。`run_mock_sample.py` は現行出力にない内部キーを比較する箇所が残っており、新規出力で `KeyError` になる。現行スキーマの検証にはREADMEのテスト一覧を使用する。

現行の関連ファイルには、初期一覧に加えて以下がある。

```text
reasoning_effort/output_schema.py
reasoning_effort/progress.py
reasoning_effort/source_context.py
migrate_reasoning_effort_output.py
migrate_reasoning_effort_directory.py
test/test_reasoning_effort_output.py
test/test_reasoning_effort_progress.py
test/test_reasoning_effort_source_context.py
```

単一ファイルの移行は `migrate_reasoning_effort_output.py`、キャッシュ・resume情報・失敗履歴まで含む移行は `migrate_reasoning_effort_directory.py` を使用する。後者は検証のみが既定で、`--apply` により新側のバックアップを作成して反映する。詳細な統合条件・固定ファイル名・再開方法はREADMEを参照。

今回の見直しはドキュメントのみであり、以下の過去のテスト結果を再実行したという意味ではない。

## 初期実装時の記録（2026-09-11）

実装日: 2026-09-11。既存実装の変更はREADME追記と依存追加のみ。
実LLMのサンプル生成は未実施（localhost:8000への接続拒否、使用先回答待ち）。
Mock生成と実Tokenizer検証は完了しており、実LLM推論完了とは区別する。

## 1–3. ファイルとディレクトリ構造

新規ファイル:

```text
main_create_reasoning_effort_dataset.py
pipelines/create_reasoning_effort_dataset.py
reasoning_effort/
  __init__.py
  cache.py
  formatters.py
  generator.py
  validators.py
prompts/create_reasoning_effort_dataset/
  thinking_low.md
  thinking_medium.md
  thinking_high.md
yamls/
  create_reasoning_effort_dataset.yaml
  create_reasoning_effort_dataset_settings_format.yaml
test/test_reasoning_effort_dataset.py
examples/reasoning_effort_dataset/
  IMPLEMENTATION_REPORT.md
  run_mock_sample.py
  sample_output/
    canonical.jsonl
    qwen3_8.jsonl
    llm_jp_4.jsonl
    summary.json
    thinking_samples.md
```

変更した既存ファイル: `README.md`, `pyproject.toml`。
追加依存: httpx、Transformers、datasets、Jinja2。既存venv・lockfileは変更していない。
通常のYAMLは既存.gitignoreで無視されるため、新規実行設定だけ `git add -f` で追加済み。その他の変更は未stage、commitはしていない。

## 4–7. Pipeline / Canonical / Mapping / 生成

Source QA → canonical effort → generator → format validation → token length validation → canonical cache → 2 formatter → 各公式template validation → 各JSONL。

内部effortはlow/medium/highのみ。Qwen highはxhigh、llm-jp highはhighへmapする。
生成モデルのAPI reasoning_effortは独立のmappingを使用する。

`expand_all`: 各effort 1回、計3回。`fixed`: 1回。
`token_length`: 指定effortプロンプトで1回生成し、reference token数で分類する。
再試行時のみ呼出数が増える。FormatterからAPIは呼ばない。

元question/answerを固定し、contextがあれば生成根拠へ渡す。元thinkingは入力しない。
元metadata・IDはsource_metadataへ保存し、派生qa_idはcanonical IDとfamilyで区別する。

Queue・worker poolで並行生成し、入力は有界windowで別threadから読み込む。
httpxのconnection pool、timeout、retryを新規generatorで設定する。

## 8–10. Validator / 空行 / 今治弁

`ThinkingFormatValidator` はタイトル、先頭節、空行、節番号連続、最低2節、本文、タグやフェンスを行単位で検査する。
エラーコードと修正指示をgeneratorへ返し、形式再試行を最大指定回数まで行う。

- タイトル直後: 空行1行。
- 各節見出し直後: 空行1行。
- 次節見出しの直前: 空行1行。
- 連続空行は禁止。

自然な今治弁を全プロンプトへ固定。既存v4の方言指針を参照し、架空根拠や無意味な水増しを禁止した。
Markdown例は `sample_output/thinking_samples.md` で人間が確認できる。

## 11–12. Token Length / Cache

reference tokenizerでcanonical_thinking_tokens、各対象Tokenizerでthinking_tokensを算出。
文字数代替はしない。effort別範囲はYAML指定。
形式・長さの両検証成功だけをcanonical journalへfsync付きで保存する。
cache keyはsource、effort、生成設定、prompt本文/version、検証条件を含む。
対象設定は含まないため、後から出力targetを追加しても再生成しない。

## 13–15. Dual Formatter

Qwen: assistant.reasoning_content / contentを公式テンプレートへ渡す。
low/medium/xhighをkwargsで条件付けし、mediumに独自directiveを入れない。

llm-jp: native messagesは公式Datasetと同じchannel/content block構造。
Transformers向けにはassistant.thinking / contentを持つtemplate_messagesを保存する。
公式DatasetではmessagesをJSON文字列として保存するが、本JSONLは配列として保持する。
テンプレートの出力はanalysis/finalのHarmonyとなり、公式parserのtoken往復も確認する。

両モデルに公式テンプレート適用済みtextとchat_template_kwargsを保存する。
thinking本文を書き換えず、元question/answerも維持する。

検証したTokenizer revision:

- Qwen: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- llm-jp: `fc7c15c262710016c19bcda4372a96ff68875846`

## 16–17. Resume / Failure

同じCLIで再実行。cacheは再検証して再利用し、target JSONLに存在するcanonical IDを成功statusとして扱う。
llm-jp highだけ失敗した試験では、再実行のgenerator呼出数0、llm-jp highだけ1件出力。
書込失敗・新target追加でもcache再利用を検証した。

生成・形式・長さ・target formatting・target writeを区別してfailures JSONLへ記録。
1件の失敗で他QAを止めず、最終失敗が残ればCLI終了コード1。
中断された末尾書込みを復旧し、中間の破損は検知する。

## 18–19. YAML / CLI

全設定は `yamls/create_reasoning_effort_dataset_settings_format.yaml` を参照。
HF Dataset、local JSONL、field mapping、generation、effort、length、tokenizer、output、async/retry設定を用意した。

```bash
python main_create_reasoning_effort_dataset.py \
  -p ./yamls/create_reasoning_effort_dataset.yaml
```

相対パスは作業ディレクトリ基準。実APIのserver_url/model_nameと必要なキー環境変数を設定する。

## 20–21. Tests / Generator Call Count

- 新規25 tests: 成功。公式Tokenizerを使う追加検証も有効化して実行。
- 既存環境の全63 tests: 成功（公式Tokenizer試験1件は未導入環境のためskip）。
- 専用環境の全63 tests: 公式Tokenizer試験も含めて全件成功、skipなし（2.993秒）。
- CLI `--help`: 成功。
- Git diff whitespace check: 成功。
- expand_all: Mock Generator 3回。
- fixed: 1回。
- formatter失敗後のresume: 0回。
- 無効thinkingはcacheにもoutputにも入らない。

テストにはmapping全値、field一致、異なるtoken計数、形式・長さretry、retry上限、異常入力継続、HTTP429、duplicate、破損cache、write failure、target追加、公式chat/Harmony validationを含む。

## 22–23. Qwen / llm-jp Sample

`sample_output/`は手書きfixtureをMock HTTP応答に使った構造検証。実LLM生成ではない。
1 QA、3 mock API要求、各モデル3件、計6件。
通常effort範囲ではなくsmoke用の1〜4096 token範囲を使用した。

| Canonical | Qwen effort | llm-jp effort | Reference tokens | Qwen tokens | llm-jp tokens |
|---|---|---|---:|---:|---:|
| low | low | low | 67 | 67 | 61 |
| medium | medium | medium | 125 | 125 | 106 |
| high | xhigh | high | 218 | 218 | 189 |

対応する全3ペアでsource_qa_id、canonical_record_id、question、thinking、answer、canonical_reasoning_effortが一致することをassertした。

## 24. Known Limitations

- 実LLMサーバーが未接続のため、実推論3回・通常token範囲でのeffort品質は未検証。依頼の全完了条件はまだ満たしていない。
- Markdown構造検証だけでは方言の自然さ、事実性、意味的な水増し、見出しなしの最終回答混入は保証できない。プロンプト制約と生成物の品質確認が必要。
- source contextは生成根拠に使い、user messageには元questionを保存する。
- ファイルロックはmacOS/Linux向けfcntl。
- 大規模データではcacheと出力journalの索引・レコードをメモリに保持する。
- 生成設定・prompt・検証条件を変えればcache keyが変わる。Tokenizer変更時は新しいtarget出力パスを使う。
- model/sourceの公開仕様はREADME記載の公式URLを参照。Tokenizerの固定revisionを更新する際は再検証する。

## 25. 既存ファイル非変更確認

開始時のGit作業ツリーはclean。
Git差分で既存Python/YAML/Prompt/Test/utilityの変更がないことを確認した。

```text
Modified existing files:
- README.md
- pyproject.toml

Other existing files modified:
- None
```

## 2026-09-12 追記: 元記事全文のID照合

`reasoning_effort/source_context.py` を追加し、QAのidと元記事idを照合する機能を実装した。
同じidの全chunkを番号順に空行で結合し、同番号ではファイル順を保持する。本文は省略しない。
前回追加したpipelineにeffort展開前の照合を組み込み、両YAMLで有効化した。
本文付きのcache・output・failureは新規パスへ分離した。元データと旧成果物は変更していない。

QAごとの照合失敗は `source_context_resolution` として1件記録し、推論を行わない。
結合した全文とsource_context_metadata（記事id、パス、chunk番号一覧、SHA-256）をcanonicalに保存し、cache keyへ反映する。
照合無効時は従来のinline contextを維持する。単独Mockサンプルは照合無効を明示した。

検証結果:

- 追加テスト12件。既存・新規・実Tokenizer検証を含む全75件が成功（skipなし）。
- 実ソース4,195行、2,326記事、chunk番号重複1件を確認。重複本文を全件保持。
- 公開QAから取得済みの60件すべてのidを照合できた。
- 実QA 1件の元記事全文36,717文字が各effortのプロンプトに完全一致で入った。
- この実QAとMock Generatorの組合せで3回呼出し、各モデル3件出力。再開は0回。
- 元記事本文変更時とcontextなし旧cacheとの分離をテストした。
- 元データの読込前後のSHA-256は一致:
  `daa22e42620a1373d62f83dbbf5014e2f950b296ceccffc3bb1635b46eb9e170`
- `git diff --check HEAD` 成功。従来のPython/YAML/Prompt/Testは変更なし。

この追加作業では実LLM推論は行っていない。長い全文を受け付ける実サーバーでの生成品質確認は引き続き未実施。

## 2026-09-12 追記: 実応答を使ったプロンプト改善

Qwen3.8-27B-NVFP4への実リクエストで、返却contentの先頭に改行2文字が付くケースを再現した。
プロンプトとsystem指示の強化だけでは改行が残った。生成用モデルの内部thinkingを無効化した比較では解消した。
このため両YAMLのgeneration.chat_template_kwargs.enable_thinkingをfalse、use_reasoning_effortをfalseへ変更した。
Canonical effortは3種類のプロンプトで引き続き制御し、Dataset側のthinkingとeffort mappingは変更していない。

- 3 promptに先頭文字・空行禁止・具体的なMarkdown例・提出前確認を追加。
- 長いcontextより優先されるsystem出力指示を追加し、cache keyへ含めた。
- 先頭空行を自動除去せず、厳密なValidatorを維持。
- retry feedbackに具体的な形式修正と実測token数・許容範囲を追加。
- request/rejected/validatedを逐次表示し、最終不合格本文をfailureのrejected_thinkingへ保存。
- 全76テスト成功、Mock HTTPサンプルも実Tokenizerで再検証。

実QA 1件と全文36,717文字による検証:

| effort | Markdown | reference tokens | Dataset出力 |
| --- | --- | ---: | --- |
| low | 合格 | 110 | 両モデルへ保存 |
| medium | 合格 | 265 | 両モデルへ保存 |
| high（最後の再試行） | 合格 | 456 | 下限768未達のため保存しない |

highは実測token数を使う再試行でも長さ下限を満たさなかった。今回のサンプルではMarkdown問題は解消したが、全effortのDataset生成成功を意味しない。長さ下限は変更していない。
成功済みlow/mediumを再開時にcacheから再利用し、highだけを再推論する動作を実サーバーでも確認した。
診断出力は `/tmp/reasoning-live-prompt-smoke/` に分離し、利用者の実行中出力には書き込んでいない。
