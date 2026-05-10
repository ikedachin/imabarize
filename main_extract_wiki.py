#!/usr/bin/env python3
"""
Wikipedia XMLダンプから今治市関連記事を抽出し、JSONL形式で保存するスクリプト。

Usage:
    uv run extract_imabari.py [--input wiki/jawiki-2026-05-01-p1p2391393.xml] [--output data/imabari/raw.jsonl] [--content-threshold 3]
"""

import argparse
import bz2
import html
import json
import re
import sys
from pathlib import Path

import lxml.etree as ET


# Wikipedia XMLのネームスペース
NS_MW = "http://www.mediawiki.org/xml/export-0.11/"

# 完全タグ用ヘルパー
def mw_tag(local: str) -> str:
    return f"{{{NS_MW}}}{local}"

# 抽出キーワード
TITLE_KEYWORD = "今治"

DROP_SECTION_TITLES = {
    "脚注",
    "注釈",
    "出典",
    "参考文献",
    "関連項目",
    "外部リンク",
    "関連文献",
    "参考資料",
}

FILE_LINK_PREFIXES = (
    "ファイル:",
    "画像:",
    "File:",
    "Image:",
    "Media:",
)

NON_CONTENT_LINK_PREFIXES = (
    "Category:",
    "カテゴリ:",
    "Help:",
    "ヘルプ:",
    "Special:",
    "特別:",
)


def is_main_namespace(ns_id: str) -> bool:
    """namespace 0（一般記事）かどうかを判定する。"""
    return ns_id == "0"


def remove_balanced_blocks(text: str, open_token: str, close_token: str) -> str:
    """入れ子を考慮してテンプレートや表を除去する。"""
    result = []
    i = 0
    depth = 0
    open_len = len(open_token)
    close_len = len(close_token)

    while i < len(text):
        if text.startswith(open_token, i):
            depth += 1
            i += open_len
            continue
        if depth and text.startswith(close_token, i):
            depth -= 1
            i += close_len
            continue
        if depth == 0:
            result.append(text[i])
        i += 1

    return "".join(result)


def drop_tail_sections(text: str) -> str:
    """CPT本文としてノイズになりやすい末尾セクションを除去する。"""
    lines = text.splitlines()
    kept = []
    dropping_level = None

    for line in lines:
        heading = re.match(r"^(={2,6})\s*(.*?)\s*\1\s*$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            if title in DROP_SECTION_TITLES:
                dropping_level = level
                continue
            if dropping_level is not None and level <= dropping_level:
                dropping_level = None

        if dropping_level is None:
            kept.append(line)

    return "\n".join(kept)


def strip_ref_tags(text: str) -> str:
    """脚注タグを除去する。自己終了タグと複数行タグの両方に対応する。"""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref\b[^/>]*/\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<ref\b[^>]*>.*?</ref\s*>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text


def find_wikilink_end(text: str, start: int) -> int:
    """start位置の内部リンクに対応する終端位置を返す。見つからない場合は-1。"""
    i = start
    depth = 0
    while i < len(text) - 1:
        if text.startswith("[[", i):
            depth += 1
            i += 2
            continue
        if text.startswith("]]", i) and depth:
            depth -= 1
            i += 2
            if depth == 0:
                return i
            continue
        i += 1
    return -1


def strip_file_links(text: str) -> str:
    """ファイル・画像リンクを、角括弧の入れ子を考慮して除去する。"""
    result = []
    i = 0

    while i < len(text):
        if text.startswith("[[", i):
            end = find_wikilink_end(text, i)
            if end != -1:
                body = text[i + 2:end - 2].lstrip(":")
                if body.startswith(FILE_LINK_PREFIXES):
                    i = end
                    continue
        result.append(text[i])
        i += 1

    return "".join(result)


def simplify_external_links(text: str) -> str:
    """外部リンクは表示ラベルだけ残す。ラベルがないURLは除去する。"""
    def replace(match: re.Match) -> str:
        body = match.group(1).strip()
        if not body:
            return ""
        parts = body.split(None, 1)
        return parts[1] if len(parts) == 2 else ""

    text = re.sub(r"\[(https?://[^\]\s]+(?:\s+[^\]]+)?)\]", replace, text)
    text = re.sub(r"https?://\S+", "", text)
    return text


def simplify_wikilinks(text: str) -> str:
    """内部リンクを表示テキストへ変換する。"""
    def body_to_text(body: str) -> str:
        body = body.lstrip(":")
        if body.startswith(NON_CONTENT_LINK_PREFIXES):
            return ""

        parts = body.split("|")
        label = parts[-1].strip()
        if not label:
            label = parts[0].strip()
        if "#" in label and label == parts[0].strip():
            label = label.split("#", 1)[-1]
        return label

    def replace(match: re.Match) -> str:
        body = match.group(1).strip()
        if not body:
            return ""
        return body_to_text(body)

    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\[\[([^\[\]]+)\]\]", replace, text)
    text = re.sub(r"\[\[([^\n]*?)\]\]", replace, text)
    return text


def normalize_lines(text: str) -> str:
    """空白、見出し、箇条書きを学習しやすい平文に整える。"""
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue
        if re.search(r"\.(?:jpg|jpeg|png|svg|gif|webp)\s*(?:\||$)", line, flags=re.IGNORECASE):
            continue

        heading = re.match(r"^={2,6}\s*(.*?)\s*={2,6}$", line)
        if heading:
            line = heading.group(1).strip()
        else:
            line = re.sub(r"^[*:;#]+\s*", "", line)
            line = re.sub(r"^\|[-+]?.*$", "", line)
            line = re.sub(r"^[!|]\s*", "", line)

        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_wiki_text(text: str) -> str:
    """
    ウィキマークアップをクリーニングし、CPT向けのプレーンテキストに変換する。
    """
    text = html.unescape(text.replace("\r\n", "\n").replace("\r", "\n"))
    text = drop_tail_sections(text)
    text = strip_ref_tags(text)
    text = strip_file_links(text)

    # 表とテンプレートはノイズ量が大きく、入れ子も多いため正規表現ではなく深さで除去する。
    text = remove_balanced_blocks(text, "{|", "|}")
    text = remove_balanced_blocks(text, "{{", "}}")

    text = simplify_external_links(text)
    text = simplify_wikilinks(text)

    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?(gallery|div|span|small|center|blockquote|poem)\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"__[^_]+__", "", text)
    text = re.sub(r"^\s*\[\[Category:.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^\s*\[\[カテゴリ:.*$", "", text, flags=re.MULTILINE)
    text = text.replace("[[", "").replace("]]", "")
    text = text.replace("{{", "").replace("}}", "")

    return normalize_lines(text)


def open_dump(input_path: str):
    """非圧縮XMLとbz2圧縮XMLの両方を開く。"""
    if str(input_path).endswith(".bz2"):
        return bz2.open(input_path, "rb")
    return open(input_path, "rb")


def parse_wiki_xml(input_path: str):
    """
    Wikipedia XMLをストリーミングでパースし、ページ情報を収束する。
    各ページは辞書のジェネレータとして返す。
    """
    with open_dump(input_path) as dump:
        context = ET.iterparse(dump, events=("end",), tag=mw_tag("page"), recover=True)
        for _, elem in context:
            yield {
                "id": elem.findtext(mw_tag("id")) or "",
                "title": elem.findtext(mw_tag("title")) or "",
                "ns": elem.findtext(mw_tag("ns")) or "0",
                "text": elem.findtext(f"{mw_tag('revision')}/{mw_tag('text')}") or "",
            }

            # メモリ解放
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]


def extract_imabari_articles(input_path: str, content_threshold: int = 3):
    """
    今治市関連記事を抽出する。

    Args:
        input_path: Wikipedia XMLファイルのパス
        content_threshold: 本文で「今治」が参照される最小回수

    Yields:
        dict: {"title": str, "content": str}
    """
    count = 0
    title_match_count = 0
    content_match_count = 0

    for page in parse_wiki_xml(input_path):
        title = page.get("title", "")
        ns = page.get("ns", "0")
        text = page.get("text", "")

        # 一般記事のみを対象
        if not is_main_namespace(ns):
            continue

        # 外部リンク
        if title.startswith(("-", ":")):
            continue
        if text.lstrip().lower().startswith(("#redirect", "#転送")):
            continue

        matched = False
        match_reason = ""

        # 基準1: タイトルに「今治」が含まれる
        if TITLE_KEYWORD in title:
            matched = True
            match_reason = "title"
            title_match_count += 1

        # 基準2: 本文で「今治」が threshold 回以上参照される
        elif text.count(TITLE_KEYWORD) >= content_threshold:
            matched = True
            match_reason = "content"
            content_match_count += 1

        if matched:
            cleaned_text = clean_wiki_text(text)
            if cleaned_text:
                count += 1
                print(f"[{count}] ({match_reason}) {title}", file=sys.stderr)
                yield {
                    "id": page.get("id", ""),
                    "title": title,
                    "content": cleaned_text
                }

    print(f"\n== 抽出完了 ==", file=sys.stderr)
    print(f"タイトル一致: {title_match_count} 件", file=sys.stderr)
    print(f"本文一致: {content_match_count} 件", file=sys.stderr)
    print(f"合計: {count} 件", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Wikipedia XMLから今治市関連記事を抽出")
    parser.add_argument(
        "--input", "-i",
        default="test_source/wiki/jawiki-****-**-**-**********.xml",
        help="Wikipedia XMLダンプのパス"
    )
    parser.add_argument(
        "--output", "-o",
        default="test_source/wiki/raw.jsonl",
        help="出力JSONLファイルのパス"
    )
    parser.add_argument(
        "--content-threshold", "-t",
        type=int,
        default=3,
        help="本文で「今治」が参照される最小回数（デフォルト: 3）"
    )
    args = parser.parse_args()

    # 入力ファイルの存在チェック
    if not Path(args.input).exists():
        print(f"エラー: 入力ファイルが見つかりません: {args.input}", file=sys.stderr)
        sys.exit(1)

    # 出力ディレクトリの作成
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"入力: {args.input}", file=sys.stderr)
    print(f"出力: {args.output}", file=sys.stderr)
    print(f"本文一致閾値: {args.content_threshold}", file=sys.stderr)
    print(f"処理中...", file=sys.stderr)

    # 抽出してJSONLに保存
    with open(output_path, "w", encoding="utf-8") as f:
        for article in extract_imabari_articles(args.input, args.content_threshold):
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    print(f"\n完了: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
