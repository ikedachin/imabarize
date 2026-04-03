# Imabarize Pipeline

標準語の日本語テキストを **今治弁** に変換するためのバッチ処理リポジトリです。  
`main_imabarize.py` がエントリポイントで、JSON / JSONL を読み込み、OpenAI互換API（OpenRouter またはローカルサーバー）経由で変換結果を JSONL で保存します。

## 機能

- JSON / JSONL の一括読み込み
- `target_key` 指定による変換対象キーの切り替え
- 複数キー同時処理（`target_key: "text,context"` のようにカンマ区切り）
- バッチ推論（`batch_size`）
- 既処理データのスキップ（`book` + `page` 単位）
- OpenRouter / ローカル OpenAI互換API の切り替え

## リポジトリ構成

- `main_imabarize.py`: 実行スクリプト
- `pipelines/imabarize_pipeline.py`: 推論・保存処理
- `prompts/imabarize.md`: 今治弁変換プロンプト
- `yamls/imabari_settings_format.yaml`: 設定テンプレート
- `test_source/`: 入力サンプル
- `test_output/`: 出力先サンプル

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

### 1) 設定ファイルを用意

テンプレートをコピーして編集:

```bash
cp yamls/imabari_settings_format.yaml yamls/imabari_settings.yaml
```

最低限の確認項目:

- `openrouter: true` の場合: `openrouter_api_key`, `openrouter_model_name`
- `openrouter: false` の場合: `SERVER_URL`, `MODEL_NAME`
- `output_path`: 出力先
- `prompts`: プロンプトファイル

注意:

- テンプレートの `prompts` が `./prompts/timabarize.md` になっている場合は、実在する `./prompts/imabarize.md` に修正してください。

### 2) 実行

ディレクトリ入力:

```bash
python main_imabarize.py \
  -s ./test_source/dummy \
  -p ./yamls/imabari_settings.yaml \
  -t context \
  -e jsonl
```

単一ファイル入力:

```bash
python main_imabarize.py \
  -s ./test_source/dummy/dummy.jsonl \
  -p ./yamls/imabari_settings.yaml \
  -t context
```

主なCLI引数:

- `-s, --source`: 入力ファイルまたはディレクトリ
- `-p, --settings_path`: YAML設定ファイル
- `-t, --target_key`: 変換対象キー（未指定時は `text`）
- `-e, --extensions`: 対象拡張子（例: `json,jsonl`）
- `-i, --start_index`: 開始インデックス（現状は設定反映のみ）

## 入出力フォーマット

### 入力（JSON / JSONL）

各レコードは辞書形式。`target_key` で指定したキーを変換対象として使用します。  
`target_key` 未指定時は `text` または `content` を探索します。

例:

```json
{"book":"sample_book","page":1,"context":"これはテストです。"}
```

### 出力（JSONL）

- 変換後テキストは **同じキー名** に上書きされます
- `generator` キーにモデル名を付与します

例:

```json
{"book":"sample_book","page":1,"context":"これは今治弁に変換された文...","generator":"qwen/..."}
```

## 再実行時のスキップ仕様

`output_path` 配下の既存 JSONL を読み、`book` と `page` が一致する入力レコードは再処理をスキップします。  
同一データに対する重複推論を避ける用途です。

## 補足

- `docs/README_main_2_imabarize.md` は旧名称（`main_2_sanitization.py`）の記述が残っています。実際の実行スクリプトは `main_imabarize.py` です。
- 本リポジトリには APIキーを含む `yamls/*.yaml` はコミットしない設定（`.gitignore`）が入っています。

## ライセンス

`LICENSE` を参照してください。




python3 main_create_imabari_qa.py -p ./yamls/imabari_settings.yaml -s ./test_source/dummy/ -t context