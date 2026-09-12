"""Durable JSONL journal, with recovery of interrupted final writes."""
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def stable_id(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


class JsonlJournal:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open("rb+") as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError(f"Non-object JSONL at {self.path}:{offset}")
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        if stream.read(1):
                            raise ValueError(f"Corrupt journal at {self.path}:{offset}")
                        stream.truncate(offset)
                        break
                    key = record.get("journal_key", record.get("canonical_record_id"))
                    if key:
                        self.records[key] = record
                    if not line.endswith(b"\n"):
                        stream.seek(0, 2)
                        stream.write(b"\n")
                        stream.flush()
                        os.fsync(stream.fileno())

    def append(self, record: dict, key: str | None = None) -> None:
        value = dict(record)
        if key:
            value["journal_key"] = key
        data = (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode()
        with self.path.open("ab") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        index = key or value.get("canonical_record_id")
        if index:
            self.records[index] = value
