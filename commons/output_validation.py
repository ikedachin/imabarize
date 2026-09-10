"""Configurable, deterministic validation of extracted output fields.

Line matching ignores surrounding whitespace and blank lines. Blank-line
requirements inspect the original LF-separated lines without normalizing them.
"""

import re
from pathlib import Path

import yaml


class OutputFormatError(ValueError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__("Output format validation failed: " + "; ".join(
            f"{error['field']}: {error['message']}" for error in errors
        ))


class OutputValidator:
    def __init__(self, config=None):
        self.fields = {}
        if config is None:
            return
        if not isinstance(config, dict):
            raise ValueError("output_validation must be a mapping")
        if not isinstance(config.get("enabled", True), bool):
            raise ValueError("output_validation.enabled must be a boolean")
        if not config.get("enabled", True):
            return
        if config.get("on_failure", "reject") != "reject":
            raise ValueError("Only output_validation.on_failure=reject is supported")
        profiles = config.get("profiles", {})
        if "profiles_file" in config:
            with Path(config["profiles_file"]).expanduser().open(encoding="utf-8") as f:
                profiles = yaml.safe_load(f)
        fields = config.get("fields", {})
        if not isinstance(profiles, dict) or not isinstance(fields, dict):
            raise ValueError("profiles and fields must be mappings")
        for field, selection in fields.items():
            if field not in {"thinking", "question", "answer"}:
                raise ValueError(f"Unsupported output field: {field}")
            if not isinstance(selection, dict):
                raise ValueError(f"Invalid profile selection for {field}")
            name = selection.get("profile")
            if not isinstance(name, str) or name not in profiles:
                raise ValueError(f"Unknown profile for {field}: {name}")
            profile = profiles[name]
            if not isinstance(profile, dict) or profile.get("type") != "text":
                raise ValueError(f"Profile {name} must have type=text")
            rules = profile.get("rules")
            if not isinstance(rules, list) or not rules:
                raise ValueError(f"Profile {name} must have nonempty rules")
            self.fields[field] = (name, [self._compile(rule) for rule in rules])

    @staticmethod
    def _compile(rule):
        if not isinstance(rule, dict):
            raise ValueError("Each validation rule must be a mapping")
        rule = dict(rule)
        kind = rule.get("type")
        allowed = {
            "line_equals": {"type", "index", "value"},
            "line_fullmatch": {"type", "index", "pattern"},
            "matching_line_count": {"type", "pattern", "min", "max"},
            "blank_line_after": {"type", "pattern"},
            "forbidden_suffix_lines": {"type", "patterns"},
        }
        if kind not in allowed or set(rule) != allowed[kind]:
            raise ValueError(f"Invalid validation rule: {rule}")
        if "index" in rule and (type(rule["index"]) is not int or rule["index"] < 0):
            raise ValueError("Line index must be a nonnegative integer")
        if kind == "line_equals" and not isinstance(rule["value"], str):
            raise ValueError("line_equals.value must be a string")
        if kind == "matching_line_count":
            if any(type(rule[key]) is not int for key in ("min", "max")) or not 0 <= rule["min"] <= rule["max"]:
                raise ValueError("Invalid matching_line_count bounds")
        patterns = rule.get("patterns", [rule["pattern"]] if "pattern" in rule else [])
        if not isinstance(patterns, list) or (kind == "forbidden_suffix_lines" and not patterns):
            raise ValueError("patterns must be a nonempty list")
        try:
            if any(not isinstance(pattern, str) for pattern in patterns):
                raise ValueError("Patterns must be strings")
            rule["compiled"] = [re.compile(pattern) for pattern in patterns]
        except re.error as exc:
            raise ValueError(f"Invalid validation regex: {exc}") from exc
        return rule

    def validate(self, outputs):
        errors = []
        for field, (profile, rules) in self.fields.items():
            value = outputs.get(field)

            def add(rule, message, line=None):
                error = dict(field=field, profile=profile, rule=rule, message=message)
                if line is not None:
                    error["line"] = line
                errors.append(error)

            if not isinstance(value, str):
                add("type", "値が文字列ではありません")
                continue
            raw = value.split("\n")
            lines = [(index, line.strip()) for index, line in enumerate(raw) if line.strip()]
            for rule in rules:
                kind = rule["type"]
                patterns = rule["compiled"]
                if kind in {"line_equals", "line_fullmatch"}:
                    index = rule["index"]
                    line = lines[index][1] if index < len(lines) else None
                    if kind == "line_equals":
                        valid = line == rule["value"]
                    else:
                        valid = line is not None and bool(patterns[0].fullmatch(line))
                    if not valid:
                        add(kind, f"空行を除く{index + 1}行目が指定形式ではありません")
                elif kind == "matching_line_count":
                    count = sum(bool(patterns[0].fullmatch(line)) for _, line in lines)
                    if not rule["min"] <= count <= rule["max"]:
                        add(kind, f"見出し等の一致行数が範囲外です: {count} (許容 {rule['min']}〜{rule['max']})")
                elif kind == "blank_line_after":
                    for index, line in lines:
                        if patterns[0].fullmatch(line):
                            # A trailing split sentinel is not an actual empty LF line.
                            if raw[index].endswith("\r") or index + 2 >= len(raw) or raw[index + 1] != "":
                                add(kind, "見出し直後に完全な空行（LFのみ）が必要です", index + 1)
                elif kind == "forbidden_suffix_lines":
                    for index, line in reversed(lines):
                        if not any(pattern.fullmatch(line) for pattern in patterns):
                            break
                        add(kind, "末尾に禁止された区切り・文字数表記・記号行があります", index + 1)
        return errors

    def check(self, outputs):
        errors = self.validate(outputs)
        if errors:
            raise OutputFormatError(errors)
