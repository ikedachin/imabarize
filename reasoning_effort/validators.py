"""Strict structural validation; never silently repair invalid model output."""
import re
from typing import Any


class ThinkingFormatValidator:
    heading = re.compile(r"^### ([1-9][0-9]*)\. (\S(?:.*\S)?)$")

    def validate(self, thinking: str) -> list[str]:
        errors: list[str] = []
        if not isinstance(thinking, str) or not thinking.strip():
            return ["empty_thinking"]
        lines = thinking.split("\n")
        if lines[0] != "## 思考プロセス":
            errors.append("first_line_mismatch")
        if len(lines) < 2 or lines[1] != "":
            errors.append("missing_blank_line_after_title")
        elif len(lines) > 2 and lines[2] == "":
            errors.append("multiple_blank_lines_after_title")
        if len(lines) < 3 or not self.heading.fullmatch(lines[2]) or not lines[2].startswith("### 1. "):
            errors.append("invalid_first_section_heading")
        if re.search(r"</?think\s*>", thinking, re.I):
            errors.append("think_tag_found")
        if "```" in thinking or "~~~" in thinking:
            errors.append("code_fence_found")
        if re.search(r"<\|[^>]+\|>|</?(?:answer|final)>", thinking, re.I):
            errors.append("final_answer_or_control_tag_found")
        if re.search(r"(?im)^\s*(?:#{1,6}\s*)?(?:final answer|最終回答|回答)\s*[:：]?\s*$", thinking):
            errors.append("final_answer_found")
        if any(a == b == "" for a, b in zip(lines, lines[1:])):
            errors.append("consecutive_blank_lines")
        if any(line and not line.strip() for line in lines):
            errors.append("whitespace_only_line")
        headings = []
        for i, line in enumerate(lines[2:], 2):
            match = self.heading.fullmatch(line)
            if match:
                headings.append((i, int(match[1])))
            elif line.lstrip().startswith("#"):
                errors.append("invalid_section_heading")
        if len(headings) < 2:
            errors.append("too_few_sections")
        for n, (index, number) in enumerate(headings):
            if number != n + 1:
                errors.append("section_number_not_contiguous")
            end = headings[n + 1][0] if n + 1 < len(headings) else len(lines)
            if index + 1 >= len(lines) or lines[index + 1] != "":
                errors.append("missing_blank_line_after_section_heading")
            elif index + 2 < len(lines) and lines[index + 2] == "":
                errors.append("multiple_blank_lines_after_section_heading")
            if n:
                if lines[index - 1] != "":
                    errors.append("missing_blank_line_before_next_section")
                elif index > 1 and lines[index - 2] == "":
                    errors.append("multiple_blank_lines_before_next_section")
            body = lines[index + 2:end]
            if not any(line.strip() and not line.lstrip().startswith("#") for line in body):
                errors.append("empty_section_body")
        return list(dict.fromkeys(errors))


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def length_errors(count: int, bounds: dict) -> list[str]:
    if count < bounds.get("min_tokens", 0):
        return ["thinking_too_short"]
    if count > bounds.get("max_tokens", float("inf")):
        return ["thinking_too_long"]
    return []


def classify_length(count: int, ranges: dict) -> str:
    matches = [effort for effort, bounds in ranges.items() if not length_errors(count, bounds)]
    if len(matches) != 1:
        raise ValueError(f"token_length_unclassified_or_ambiguous: {count}")
    return matches[0]


def cleanup(thinking: str) -> str:
    # Normalize line endings and one conventional terminal newline only.
    # Leading whitespace, wrappers and broken Markdown must reach the validator.
    return thinking.replace("\r\n", "\n").removesuffix("\n")


def retry_feedback(errors: list[str]) -> str:
    details = {
        "first_line_mismatch": "先頭に空行や空白を置かず、応答の最初の文字を # にして、1行目を ## 思考プロセス にしてください。",
        "invalid_first_section_heading": "3行目を ### 1. 質問の整理 にしてください。",
        "consecutive_blank_lines": "連続する空行は禁止です。区切りは改行文字2個（空行1行）だけにしてください。",
        "section_number_not_contiguous": "節番号は1, 2, 3の順に連続させてください。",
    }
    return (
        "前回の出力は条件違反です: " + ", ".join(errors)
        + "。" + " ".join(details[e] for e in errors if e in details)
        + "。タイトル直後、各見出し直後、次の見出しの直前には空行をちょうど1行入れてください。"
        "番号は1から連続、最低2節、各節に本文が必要です。タグや最終回答は含めず、"
        "指定token範囲内で自然な今治弁のthinking本文だけを再生成してください。"
    )
