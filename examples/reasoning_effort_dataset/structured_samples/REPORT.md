# 構造化messages出力への変更報告

現行の新形式サンプルは、このディレクトリの `qwen38.jsonl` と `llmjp4.jsonl` です。`../sample_output/` は旧形式の履歴資料です。以下の変更ファイルと86件のテスト結果は実装時の記録で、ドキュメント見直し時に再実行した結果ではありません。

## 変更ファイル

- `reasoning_effort/formatters.py`: messagesを直接生成。template_messages生成を削除。展開文字列は既存の整合性検証に一時使用するだけとし、textを保持しない。chat_template_kwargsはllm-jpだけ保持。Qwenはeffortだけをテンプレートへ渡す。既存のcanonical→モデル別ラベル対応を明記し、不明ラベルを拒否。
- `reasoning_effort/output_schema.py`: モデル別許可リストとmessagesの厳密な一致検証。llm-jpのeffort・固定日付を保持。Qwen出力のhighを拒否。
- `migrate_reasoning_effort_output.py`: 新旧スキーマ間の移行に対応。旧Harmony形式だけは旧template_messagesを読み取って移行する。新規生成・新規出力にはこのキーは存在しない。
- `test/test_reasoning_effort_dataset.py`: 公式Tokenizerによる全effortの互換性、Qwenのデフォルト動作、禁止キーの不在を検証。
- `test/test_reasoning_effort_output.py`: スキーマ、effort対応、不正なhighやmessagesの拒否、新旧移行と再開を検証。
- `README.md`: 新スキーマ、学習時のテンプレート適用、移行手順を更新。
- このディレクトリの `qwen38.jsonl` と `llmjp4.jsonl`: 各１行の固定サンプル。thinkingはテストfixtureで、token数は各公式Tokenizerで計算。実LLM生成は行っていない。

## 最終スキーマ

Qwen（13キー、記載順）:

```text
qa_id, source_qa_id, id, chunk_index, question, answer, thinking,
reasoning_effort, eval, messages, thinking_tokens, thinking_generator,
source_metadata
```

LLM-jp（14キー）: 上記messagesの直後にchat_template_kwargsを追加。

- Qwen assistant: role, content, reasoning_content
- LLM-jp assistant: role, content, thinking
- 両モデルのuser: role, content
- source_metadata: qa_id, id, chunk_index
- llm-jp chat_template_kwargs: reasoning_effort, conversation_start_date

question、answer、thinkingとmessagesの値が完全一致しないレコードは拒否する。両モデルともtext・template_messagesは保存しない。Qwenはchat_template_kwargsも保存しない。

## effort

既存設定と実装は共通生成にlow/medium/highを使用している。既存の明示的対応表は維持する。

| 共通生成 | Qwen出力 | llm-jp出力 |
|---|---|---|
| low | low | low |
| medium | medium | medium |
| high | xhigh | high |

Qwenの既存出力レコードや移行入力にhighが指定されている場合は、xhighに補正せずエラー。canonical側のxhighも不正ラベルとして拒否。生成ラベルの仕様、プロンプト、生成キャッシュキーを変えていない。

## 公式資料と検証

- [Qwen公式テンプレート](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/chat_template.jinja)
- [llm-jp 32B公式テンプレート](https://huggingface.co/llm-jp/llm-jp-4-32b-a3b-thinking/blob/main/chat_template.jinja)
- [Transformers公式説明](https://huggingface.co/docs/transformers/chat_templating)

実行した回帰テスト: `python -m unittest discover -s test`（REASONING_REAL_TOKENIZERSをローカル公式Tokenizerキャッシュに設定）。86件すべて成功、スキップなし。

公式Qwenと既存設定のllm-jp 8B Tokenizerで全３effortを検証。Qwenはeffortだけを渡した結果とenable_thinking/preserve_thinkingを明示的にtrueにした結果が完全一致。llm-jpはanalysis/finalの本文とtoken往復を検証した。取得した最新32Bテンプレートも既存Tokenizerのapply_chat_templateへ明示的に渡し、全３effortのanalysis/final本文の一致を確認した。32B Tokenizerへの切替やモデル学習は行っていない。

検証途中で既存test_output/reasoning_effort_with_contextが見つからなくなったため、実ファイル全行の変換前後比較は未完了。こちらから既存データの削除・移動・上書きは行っていない。固定入力を用いた構造検証・再開・移行・欠損・失敗テストを実施した。

## 構造化messagesへの変更時に維持した処理

QA・回答・thinking生成、今治弁プロンプト、Markdown検証、eval、ID生成、metadataの意味、モデル別token計算、並列/非同期/API/リトライ、キャッシュ・再開、既存ファイルの命名・既存成果物。既存成功行は再開しても自動変換されない。

既存ファイルをそろえる場合は、生成終了後に移行コードへ入力JSONLと別の出力パスを指定する。新形式入力の再開情報は隣接する.resume.jsonlから取得する。


## 2026-09-14 追記: ディレクトリ移行と全件確認

上記の「実ファイル全行の変換前後比較は未完了」は、構造化messages変更時点の状況。その後、`migrate_reasoning_effort_directory.py` を追加し、次の移行先について全件確認を完了した。

- 旧側: `test_output/reasoning_effort_with_context_old`
- 新側: `test_output/reasoning_effort_with_context`
- 反映前バックアップ: 新側の `migration-backup-20260914-224125-x0tk02r8/`

| 対象 | 移行確認時の件数 | 確認内容 |
| --- | ---: | --- |
| Qwen出力 | 5,702 | 全件が軽量スキーマに一致 |
| llm-jp出力 | 5,702 | 全件が軽量スキーマに一致 |
| 各resumeファイル | 各5,702 | 出力・キャッシュのcanonical IDと一致、重複なし |
| 共通キャッシュ | 5,702 | `generation_context` 除去、キー保持、Markdown形式エラーなし |
| 失敗履歴 | 20,743 | 旧側と新側の履歴を統合し、完全一致の重複を除去 |

旧側と反映前バックアップから期待結果を再計算し、移行先の全6ファイルのJSONレコードが一致することを確認した。出力とキャッシュの `source_qa_id`・質問・回答・thinkingも全件一致した。モデル別effort件数はlow 3,091件、medium 2,319件、high相当292件（Qwenはxhigh）。これは移行直後の記録であり、今後の生成・再実行で件数は変わる。

この全件確認は形式・保存内容・ID整合性の検証であり、Tokenizerによるtoken数の再計測、実LLM推論、方言や事実性の全件評価は行っていない。元ディレクトリとバックアップには復元用の旧形式が残る。

移行コマンドは [README](../../../README.md) を参照。ディレクトリ移行では同じcanonical IDの新側データを優先し、キャッシュキーを再計算しない。生成設定やプロンプトを変えた場合の再利用まで保証するものではない。
