# QA出力の形式検査

`main_create_imabari_qa.py`の設定に以下を指定します。
標準の`yamls/create_imabari_qa_settings_format.yaml`では有効です。
既存の個別設定にも、このブロックを追加すると有効になります。
設定がない場合、または`enabled: false`の場合は従来の動作です。

```yaml
output_validation:
  enabled: true
  on_failure: reject
  profiles_file: ./yamls/qa_output_validation_format.yaml
  fields:
    thinking:
      profile: thinking_markdown_v2
```

パスは既存のプロンプト指定と同様、実行時の作業ディレクトリ基準です。
`profiles_file`を省略し、`profiles`に同じ定義を直接記述することもできます。
両方を指定した場合はファイルの定義を使用します。

## 標準のthinking検査

- 最初の空行でない行が`## 思考プロセス`。
- 次の空行でない行が`### 番号. 見出し名`。
- 番号付き見出しの一致行が1〜5個。番号の連続性や見出し名は指定しません。
- タイトルとすべての番号付き見出しの直後に、完全な空行が1行以上必要。
  見出し行の改行と直後の空行はLFを要求します。スペース・タブ入りの空行や
  CRLFは不合格です。検査では改行を変換しません。
- 末尾のコードフェンス、引用符・記号だけの行、文字数表記を禁止。
  末尾から空白行を飛ばし、禁止パターンに一致する行が連続する間を調べます。

見出しの一致判定は各行の前後の空白を無視しますが、直後の空行の検査は
元の文字列を使います。検査対象は既存のタグ抽出処理を経た各フィールドです。
検査処理自身は出力を整形・削除しません。

末尾の検査は行全体の正規表現一致です。本文中の引用符や通常の文章に含まれる
数字は禁止しません。任意の「無意味な文章」の判別、途中のコードブロックの
禁止、本文の正確さや方言の評価は行いません。
実際の不合格例に応じて`forbidden_suffix_lines.patterns`を追加してください。

## 実行位置と失敗記録

Step 4でthinking・answerが修正された後、Step 5の評価リクエスト前に検査します。
修正プロンプトがない場合も、保持している最終値を検査します。
不合格のレコードには評価リクエストを送りません。

通常出力・処理済みキャッシュへの登録を行わず、既存の`*.failures.jsonl`へ
保存します。`error_type: output_format`、`validation_errors`に加えて、
`previous_outputs`に検査対象の生成文を保持します。`failed_step`は検査を実行する
Step 5です。`id`・`chunk_index`・`qa_id`は既存の保存処理で付与されます。
バッチ終了時にフィールド・ルール別の不合格レコード数をログ出力します。
同じレコードで同じルールに複数回違反しても、その集計では1件です。
自動修正や形式違反の自動リトライは行いません。

## フィールドとルールの追加

`fields`には`thinking`、`question`、`answer`を指定できます。
プロファイルは`type: text`と`rules`のリストで定義します。

| ルール | 必須パラメータ | 判定 |
| --- | --- | --- |
| `line_equals` | `index`, `value` | 空行を除く0始まりの指定行の一致 |
| `line_fullmatch` | `index`, `pattern` | 同じ指定行の正規表現完全一致 |
| `matching_line_count` | `pattern`, `min`, `max` | 正規表現に完全一致する行数 |
| `blank_line_after` | `pattern` | 一致行の直後にLFだけの空行 |
| `forbidden_suffix_lines` | `patterns` | 末尾にある禁止行の検出 |

例えば、answerを1行に制限するプロファイルは次のように追加できます。

```yaml
answer_single_line_v1:
  type: text
  rules:
    - type: matching_line_count
      pattern: '.+'
      min: 1
      max: 1
```

その上で`fields.answer.profile: answer_single_line_v1`を設定します。
未知のプロファイル、未対応のルール、不正な正規表現等は起動時にエラーにします。
プロンプト側でも、選んだプロファイルと一致する形式を要求してください。
標準テンプレートが参照する`prompts/create_qa`には今回の形式要件を反映しています。

既存JSONLの再判定にも同じ検証器を利用できます（この呼び出しはファイルを変更しません）。

```python
from commons.output_validation import OutputValidator

validator = OutputValidator(settings["output_validation"])
errors = validator.validate(record)
```

## テスト

```bash
.venv/bin/python -m unittest test.test_output_validation test.test_create_qa_model
```

外部APIを使わずに、空行・見出し数・末尾の境界条件、修正後の検査、
失敗JSONLとキャッシュ、既存のQAパイプライン動作を確認します。
