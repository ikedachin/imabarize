# Imabarize

![Imabarize](./imabarize.png)

画像はnanobananaです。

Wikipediaや既存の文章から、今治弁のQ&A、継続事前学習用のCPT、評価用・GRPO用・Reasoning Effort別のデータセットを作成し、Hugging Faceへアップロードするためのツール集です。生成にはOpenAI互換API（OpenRouter / ローカルサーバー）を使用します。Wikipedia抽出と、再構成処理を無効にしたCPT生成は生成APIを使いません。

## 用途別README

| やりたいこと | README | 対象プログラム |
| --- | --- | --- |
| 原文の抽出、通常のCPT・Q&Aを生成する | [メイン処理](docs/main/README.md) | `main_extract_wiki.py`、`main_create_cpt_dataset.py`、`main_create_imabari_qa.py` |
| 生成済みデータをHugging Faceへ公開する | [アップロード](docs/upload/README.md) | `main_upload_dataset.py`、`tool_upload_for_llmjp4.py`、アップロード用Notebook 2本 |
| 評価・GRPO・Reasoning Effortのデータを作る、旧形式を移行する | [特殊なデータセット](docs/special_datasets/README.md) | `main_create_eval_qa.py`、`main_judge_eval_qa.py`、`main_create_grpo_qa.py`、`main_create_reasoning_effort_dataset.py`、`migrate_*.py`、`tool_delete_generation_context.py` |

各READMEに目的・入力と出力・設定・参考コマンドをまとめています。従来の生成設定・入出力形式・Reasoning Effortの詳細説明も用途別READMEに移しました。

## セットアップ

Python 3.11以上と `uv` を使用します。リポジトリルートで依存関係を準備してください。

```bash
uv sync
```

以降のコマンドもすべてリポジトリルートで実行します。YAML内の相対パスは作業ディレクトリ基準です。`uv` を使わない場合は、仮想環境に `pyproject.toml` の依存関係を導入して `python` で実行してください。

- `yamls/*_format.yaml` が共有設定テンプレートです。用途別READMEの例に従ってコピーし、入力・出力先、モデル名、API接続先を編集します。既存のローカル設定がある場合はその内容を確認して使用してください。
- 通常のQ&A・CPT・評価・GRPOは、YAMLの `openrouter` と接続設定を使います。Reasoning Effortは独立した `generator` 設定と、`generator.api_key_env` が指す環境変数を使います。
- Hugging Face認証はアップロードツールごとに異なるため、[アップロード用README](docs/upload/README.md)を参照してください。
- `test_source/` の入力、`test_output/` の生成結果、ローカル設定の多くはGit管理対象外です。コマンド例のデータは別途用意してください。

## 最初の実行例

設定をコピーし、API接続先・モデル・`output_path` を編集してから実行します。`cp -n` は既存の設定を上書きしません。以下は入力JSONLの `content` からQ&Aを作る例です。

```bash
cp -n yamls/create_imabari_qa_settings_format.yaml yamls/create_imabari_qa_settings.yaml

uv run python main_create_imabari_qa.py \
  -s ./test_source/wiki/raw.jsonl \
  -p ./yamls/create_imabari_qa_settings.yaml \
  -t content
```

原文の準備から進める場合は[メイン処理](docs/main/README.md)、生成済みQ&Aからthinkingのeffort違いを作る場合は[特殊なデータセット](docs/special_datasets/README.md)へ進んでください。

## リポジトリ構成

| 場所 | 内容 |
| --- | --- |
| `main_*.py` | 生成・評価・抽出・通常アップロードのCLI |
| `tool_*.py` / `tool_*.ipynb` | 特定スキーマ向けアップロード・データ整形 |
| `migrate_*.py` | 既存Reasoning Effortデータの形式移行 |
| `pipelines/` | 非同期生成・採点パイプラインの実装 |
| `reasoning_effort/` | thinking生成・検証・モデル別整形・キャッシュ・進捗表示 |
| `commons/` | 共通の設定・入力・メッセージ等の処理 |
| `prompts/` / `yamls/` | プロンプトと設定テンプレート |
| `docs/` | 用途別README |
| `examples/` | 出力例・検証資料（旧形式の例も含む） |
| `test/` | 自動テスト |
| `test_source/` / `test_output/` | ローカル入力・生成結果 |

## 成果物の例
### CPT set
- [imabari_wiki_cpt_v3](https://huggingface.co/datasets/ikedachin/imabari_wiki_cpt_v3)


### QA set
- [JaQuAD_imabari_v1](https://huggingface.co/datasets/ikedachin/JaQuAD_imabari_v1)
- [JaQuAD_imabari_v2](https://huggingface.co/datasets/ikedachin/JaQuAD_imabari_v2)
- [imabari_wiki_qa_v3](https://huggingface.co/datasets/ikedachin/imabari_wiki_qa_v3)
- [imabari_wiki_qa_v4](https://huggingface.co/datasets/ikedachin/imabari_wiki_qa_v4)

## 参考

※このリポジトリは`https://github.com/foxn2000/sdg`にインスパイアされたレポジトリです。sdgレポジトリはさらに進化し、`foxn2000
sdg_loom`に進化しています。こちらも是非ともご覧ください。

## ライセンス

Apache License 2.0。[LICENSE](LICENSE) を参照してください。
