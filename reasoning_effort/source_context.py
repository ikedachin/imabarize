"""Read-only article lookup: retain every chunk and never truncate source text."""
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def normalize_id(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value == "":
        raise ValueError("missing_or_invalid_source_id")
    return str(value)


@dataclass(frozen=True)
class ArticleContext:
    article_id: str
    text: str
    chunk_indices: tuple[int, ...]
    text_sha256: str


class SourceContextIndex:
    def __init__(self, config: dict):
        if config.get("mode", "concatenate_all") != "concatenate_all":
            raise ValueError("source_context.mode must be concatenate_all")
        self.path = Path(config["path"]).expanduser().resolve()
        self.qa_id_field = config.get("qa_id_field", "id")
        groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
        self.invalid_articles: dict[str, str] = {}
        self.row_count = 0
        self.duplicate_chunk_count = 0
        seen = set()
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("source row must be an object")
                    article_id = normalize_id(row.get(config.get("source_id_field", "id")))
                except ValueError as exc:
                    raise ValueError(f"Invalid source context at {self.path}:{line_number}: {exc}") from exc
                self.row_count += 1
                text = row.get(config.get("text_field", "text"))
                chunk = row.get(config.get("chunk_index_field", "chunk_index"))
                if not isinstance(text, str) or not text.strip():
                    self.invalid_articles[article_id] = "missing_or_empty_source_text"
                    continue
                if isinstance(chunk, bool) or not isinstance(chunk, int):
                    self.invalid_articles[article_id] = "missing_or_invalid_chunk_index"
                    continue
                key = (article_id, chunk)
                if key in seen:
                    self.duplicate_chunk_count += 1
                    logger.warning("Duplicate source chunk retained: id=%s chunk_index=%s line=%s",
                                   article_id, chunk, line_number)
                seen.add(key)
                groups[article_id].append((chunk, text))
        self.articles: dict[str, ArticleContext] = {}
        for article_id, chunks in groups.items():
            if article_id in self.invalid_articles:
                continue
            # Python's stable sort preserves file order for equal chunk indices.
            chunks.sort(key=lambda item: item[0])
            text = "\n\n".join(body for _, body in chunks)
            self.articles[article_id] = ArticleContext(
                article_id, text, tuple(index for index, _ in chunks),
                hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )

    def resolve(self, qa: dict) -> ArticleContext:
        article_id = normalize_id(qa.get(self.qa_id_field))
        if article_id in self.invalid_articles:
            raise ValueError(self.invalid_articles[article_id])
        if article_id not in self.articles:
            raise ValueError(f"source_id_not_found: {article_id}")
        return self.articles[article_id]
