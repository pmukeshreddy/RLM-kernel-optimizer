from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from .env_loader import load_project_env

logger = logging.getLogger(__name__)

try:
    from pinecone import Pinecone
except ImportError:  # pragma: no cover - optional dependency
    Pinecone = None


@dataclass
class PineconeMatch:
    match_id: str
    score: float
    text: str
    title: str = ""
    source: str = ""
    metadata: dict | None = None

    def to_legacy_dict(self) -> dict:
        return {
            "path": self.source,
            "content": self.text,
            "title": self.title or self.match_id,
            "metadata": self.metadata or {},
            "score": self.score,
        }


class PineconeRetriever:
    def __init__(self, config: dict | None = None):
        load_project_env()
        cfg = config or {}
        self.enabled = str(cfg.get("provider", "pinecone")).lower() == "pinecone"
        self.top_k = int(cfg.get("top_k", 4))
        self.namespace = cfg.get("namespace") or os.getenv(
            cfg.get("namespace_env", "PINECONE_NAMESPACE")
        )
        self.fields = list(cfg.get("fields", ["chunk_text", "title", "source"]))
        self.text_field = cfg.get("text_field", "chunk_text")
        self.title_field = cfg.get("title_field", "title")
        self.source_field = cfg.get("source_field", "source")
        self.api_key_env = cfg.get("api_key_env", "PINECONE_API_KEY")
        self.index_host_env = cfg.get("index_host_env", "PINECONE_INDEX_HOST")
        self.default_filter = cfg.get("metadata_filter") or None

        self._client = None
        self._index = None
        self._init_error = ""

    def _ensure_index(self):
        if not self.enabled:
            self._init_error = "RAG provider is disabled."
            return None

        if self._index is not None:
            return self._index
        if self._init_error:
            return None
        if Pinecone is None:
            self._init_error = "The pinecone package is not installed."
            return None

        api_key = os.getenv(self.api_key_env)
        index_host = os.getenv(self.index_host_env)
        if not api_key or not index_host:
            self._init_error = (
                f"Missing environment variables: {self.api_key_env} and/or "
                f"{self.index_host_env}."
            )
            return None

        try:
            self._client = Pinecone(api_key=api_key)
            self._index = self._client.Index(host=index_host)
        except Exception as exc:  # pragma: no cover - network runtime
            self._init_error = f"Failed to initialize Pinecone index: {exc}"
            logger.warning(self._init_error)
            return None

        return self._index

    def status(self) -> str:
        if self._ensure_index() is not None:
            return "configured"
        return self._init_error or "not configured"

    def search(
        self,
        query: str,
        top_k: int | None = None,
        namespace: str | None = None,
        metadata_filter: dict | None = None,
    ) -> list[PineconeMatch]:
        if not query.strip():
            return []

        index = self._ensure_index()
        if index is None:
            return []

        effective_filter = metadata_filter if metadata_filter is not None else self.default_filter
        search_query = {
            "inputs": {"text": query.strip()},
            "top_k": int(top_k or self.top_k),
        }
        if effective_filter:
            search_query["filter"] = effective_filter

        try:
            response = index.search(
                namespace=namespace or self.namespace,
                query=search_query,
                fields=self.fields,
            )
        except Exception as exc:  # pragma: no cover - network runtime
            logger.warning("Pinecone search failed for query %r: %s", query, exc)
            return []

        hits = self._extract_hits(response)
        matches = []
        for hit in hits:
            fields = hit.get("fields", {}) or {}
            metadata = hit.get("metadata", {}) or {}
            text = str(fields.get(self.text_field) or metadata.get(self.text_field) or "").strip()
            if not text:
                continue
            matches.append(
                PineconeMatch(
                    match_id=str(hit.get("_id") or hit.get("id") or "unknown"),
                    score=float(hit.get("_score") or hit.get("score") or 0.0),
                    text=text,
                    title=str(
                        fields.get(self.title_field)
                        or metadata.get(self.title_field)
                        or hit.get("_id")
                        or hit.get("id")
                        or ""
                    ).strip(),
                    source=str(
                        fields.get(self.source_field)
                        or metadata.get(self.source_field)
                        or ""
                    ).strip(),
                    metadata=metadata,
                )
            )

        return matches

    def search_many(
        self,
        queries: list[str],
        top_k: int | None = None,
        namespace: str | None = None,
        metadata_filter: dict | None = None,
    ) -> list[PineconeMatch]:
        deduped = {}
        for query in queries:
            for match in self.search(
                query=query,
                top_k=top_k,
                namespace=namespace,
                metadata_filter=metadata_filter,
            ):
                existing = deduped.get(match.match_id)
                if existing is None or match.score > existing.score:
                    deduped[match.match_id] = match
        return sorted(deduped.values(), key=lambda item: item.score, reverse=True)

    def format_matches(self, matches: list[PineconeMatch], max_chars: int = 4000) -> str:
        if not matches:
            return f"No Pinecone matches. Retriever status: {self.status()}"

        parts = []
        total_chars = 0
        for idx, match in enumerate(matches, start=1):
            header = f"[{idx}] {match.title or match.match_id}"
            if match.source:
                header += f" | {match.source}"

            body = match.text.strip().replace("\r", "")
            block = f"{header}\nScore: {match.score:.3f}\n{body}"
            total_chars += len(block)
            if total_chars > max_chars:
                break
            parts.append(block)

        return "\n\n".join(parts)

    def get_top_k(self, query: str, k: int = 1):
        return [match.to_legacy_dict() for match in self.search(query, top_k=k)]

    def _extract_hits(self, response) -> list[dict]:
        if isinstance(response, dict):
            result = response.get("result", response)
            hits = result.get("hits", [])
            return [self._normalize_hit(hit) for hit in hits]

        result = getattr(response, "result", response)
        hits = getattr(result, "hits", [])
        return [self._normalize_hit(hit) for hit in hits]

    def _normalize_hit(self, hit) -> dict:
        if isinstance(hit, dict):
            return hit

        output = {}
        for key in ("_id", "id", "_score", "score", "fields", "metadata"):
            value = getattr(hit, key, None)
            if value is not None:
                output[key] = value
        if hasattr(hit, "to_dict"):
            try:
                output.update(hit.to_dict())
            except Exception:
                pass
        return output


def init_knowledge_base(config: dict | None = None) -> PineconeRetriever:
    return PineconeRetriever(config=config)


if __name__ == "__main__":
    retriever = init_knowledge_base()
    demo_queries = [
        "Blackwell CUDA vectorized stores",
        "warp reduction bf16 kernel",
    ]
    print(json.dumps({"status": retriever.status(), "queries": demo_queries}, indent=2))
    print(retriever.format_matches(retriever.search_many(demo_queries)))
