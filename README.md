# 🚀 SFT data builder

**Supervised Fine-Tuning Data Generation Pipeline**

<div align="center">

![COSMIC SFT Engine](assets/sftdata_builder.png)

*PDF/テキスト/JSON から以下を段階的に生成するパイプライン*

**[📄 OCR]** → **[🔄 サニタイズ]** → **[💬 Q&A生成]**

</div>

---

## 📌 概要

このプロジェクトは、PDF/テキスト/JSON などの非構造化データから **SFT(Supervised Fine-Tuning)** に最適化されたデータセットを生成するための包括的なパイプラインです。くれぐれも元データ（PDF/テキスト/JSONなど）の著作権、ライセンスを遵守してください。

`docs/` に各工程の詳細があり、この README はリポジトリ全体の実行導線をまとめたものです。

## ✨ 主な機能

| 機能 | 説明 |
|------|------|
| **OCR処理** | PDFなどの資料を学習可能なテキストデータに変換。RAGや継続事前学習に必要なデータを生成 |
| **サニタイズ処理** | テキストの版権処理・品質向上。翻訳ベースのデータクリーニング |
| **Q&A生成** | 抽出したテキストから SFT用のQ&Aデータセットを自動生成 |

## 📊 処理フロー

```
┌─────────────────────────────────────────────┐
│ 📁 入力データ (PDF/JSON/テキスト等)            │
└────────────────┬────────────────────────────┘
                 │
         ┌───────▼─────────────┐
         │ 1️⃣  OCR処理        │
         │ main_1_ocr.py       │
         │ ↓ JSONL形式出力     │
         └───────┬─────────────┘
                 │
         ┌───────▼──────────────────┐
         │ 2️⃣  サニタイズ         │
         │ main_2_sanitization.py   │
         │ ↓ 品質改善済みデータ      │
         └───────┬──────────────────┘
                 │
         ┌───────▼────────────┐
         │ 3️⃣  Q&A生成       │
         │ main_3_create_qa.py│
         │ ↓ SFT学習データ    │
         └───────┬────────────┘
                 │
         ┌───────▼──────────┐
         │ 📤 出力 JSONL     │
         └───────────────────┘
```

> 💡 **注:** テキスト化・JSON化済みデータがある場合は `main_2_sanitization.py` から開始可能

## このリポジトリの目的

このプロジェクトは、次の目的を達成するためのパイプラインです。

1. **OCRを実行してPDFをデータ化**  
   PDFなどの資料を学習可能なテキストデータに変換。RAGや継続事前学習に必要なデータを生成します。

2. **版権処理・品質改善**  
   許可を得て OCR したデータに対して、サニタイズ工程で必要な処理を行います。

3. **SFT用のQ&Aセット作成**  
   抽出したテキストデータから、SFT用のデータセットを自動生成します。

4. **思考プロセスの追加（特殊事例）**  
   既存の QA データセットに思考過程を動的に追加します。

## 全体フロー

通常は次の順で実行します。

1. `main_1_ocr.py`  
   PDF → OCR JSONL（`test_output/ocr/`）
2. `main_2_sanitization.py`  
   JSON/JSONL → sanitized JSONL（`test_output/sanitization_test/`）
3. `main_3_create_qa.py`  
   テキスト/JSON/JSONL → QA JSONL（`test_output/test_qa/`）

既にテキスト化・JSON化されたデータがある場合は `main_2_sanitization.py` から開始できます。

## セットアップ

**前提:**
- Python 3.11+
- OpenRouter もしくは OpenAI互換ローカル推論サーバー

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

## クイックスタート

### A. 1 → 2 → 3 を通しで実行

```bash
# 1) OCR
python main_1_ocr.py \
  -s ./test_source/pdfs \
  -p ./yamls/ocr_settings_format.yaml

# 2) Sanitization（main_1出力を入力）
python main_2_sanitization.py \
  -s ./test_output/ocr \
  -p ./yamls/sanitization_settings_format.yaml \
  -t text

# 3) Q&A（main_2出力を入力）
# main_2でtarget_key未指定の場合は sanitized_text
python main_3_create_qa.py \
  -s ./test_output/sanitization_test \
  -t sanitized_text \
  -p ./yamls/create_qa_settings_format.yaml
```

### B. 既存JSON/JSONLがある場合（main_2から開始）

```bash
# 2) Sanitization（test_sourceの既存JSONLから）
python main_2_sanitization.py \
  -s ./test_source/jsonls/sample_sanitized.jsonl \
  -t original_text \
  -p ./yamls/sanitization_settings_format.yaml

# 3) Q&A（main_2で -t original_text を使ったので sanitized_original_text を指定）
python main_3_create_qa.py \
  -s ./test_output/sanitization_test \
  -t sanitized_original_text \
  -p ./yamls/create_qa_settings.yaml
```

## 入出力サンプル

**入力 (`test_source/`):**
- `test_source/pdfs/` — OCR入力
- `test_source/jsons/`, `test_source/jsonls/` — Sanitization/Q&A入力
- `test_source/texts/`, `test_source/mds/` — Q&A入力

**出力 (`test_output/`):**
- `test_output/ocr/` — OCR処理結果
- `test_output/sanitization_test/` — サニタイズ済みデータ
- `test_output/test_qa/` — Q&Aデータセット

## 主要スクリプト

- **`main_1_ocr.py`**  
  PDF を画像化し、OCR・オブジェクト検出・内容理解・重複チェックを実行  
  出力: `<pdf_stem>.jsonl`

- **`main_2_sanitization.py`**  
  入力テキストを翻訳ベースでサニタイズ  
  出力キー: `sanitized_<target_key>`, `similarity_<target_key>`, `eval_<target_key>`  
  出力: `sanitized_<book>.jsonl`

- **`main_3_create_qa.py`**  
  テキスト/JSON/JSONL から Q&A を生成  
  出力: `<parent>.jsonl`（テキスト入力時）または `<input_stem>.jsonl`（JSON入力時）


## 設定ファイル

各スクリプトの設定は `yamls/` の設定ファイルを使用します。
`*_format.yaml`をコピーして使ってください。

- `yamls/ocr_settings.yaml`
- `yamls/sanitization_settings_format.yaml`
- `yamls/create_qa_settings.yaml`

### 主な設定項目

**推論接続:**
- `openrouter` — OpenRouter API を使用
- `openrouter_api_key` — OpenRouter APIキー
- `openrouter_server_url` — サーバーURL
- `openrouter_model_name` — モデル名

**ローカル推論:**
- `SERVER_URL` — ローカル推論サーバーURL
- `MODEL_NAME` — モデル名

**共通:**
- `infer_config` — 推論設定
- `batch_size` — バッチサイズ
- `max_retries` — 最大リトライ回数
- `wait_seconds` — 待機秒数
- `output_path` — 出力パス

## プロンプト

各プログラムで使用するプロンプトは `./prompts/` フォルダ内にあります。

> ⚠️ **重要:** 現在のプロンプトは医療分野を想定したサンプルです。実運用では、対象ドメイン（例: 法務、金融、製造、教育など）に合わせて `prompts/` 配下の文面・評価観点・用語を必ず書き換えてください。

推奨手順:
1. 対象ドメインの専門用語・禁止事項・出力要件を整理する
2. `prompts/ocr/` `prompts/sanitization/` `prompts/create_qa/` の各プロンプトをドメイン向けに調整する
3. `test_source/` の少量データで試験実行し、出力品質を確認してから本番処理する

## ドキュメント

詳細は `docs/` を参照してください。

- [環境セットアップ](docs/README_environment.md)
- [main_1_ocr.py](docs/README_main_1_ocr.md)
- [main_2_sanitization.py](docs/README_main_2_sanitization.md)
- [main_3_create_qa.py](docs/README_main_3_create_qa.md)

---

<div align="center">

**Made with ❤️ for SFT Data Generation**

</div>
