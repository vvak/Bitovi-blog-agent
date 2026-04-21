from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

import ingest


def test_load_blog_documents_reads_cache_without_scraping(
    monkeypatch,
    tmp_path: Path,
    blog_documents: list[Document],
) -> None:
    cache_path = tmp_path / "blog_documents.json"
    ingest._write_document_cache(blog_documents, cache_path)
    monkeypatch.setattr(ingest, "DOCUMENT_CACHE_PATH", cache_path)

    def fail_scrape() -> list[Document]:
        raise AssertionError("cached ingest should not scrape")

    monkeypatch.setattr(ingest, "scrape_blog_documents", fail_scrape)

    documents = ingest.load_blog_documents()

    assert documents == blog_documents


def test_load_blog_documents_refreshes_cache_when_requested(
    monkeypatch,
    tmp_path: Path,
    blog_documents: list[Document],
) -> None:
    cache_path = tmp_path / "blog_documents.json"
    stale_documents = [
        Document(
            page_content="Stale content",
            metadata={
                "source": "https://www.bitovi.com/blog/stale",
                "title": "Stale",
            },
        )
    ]
    ingest._write_document_cache(stale_documents, cache_path)
    monkeypatch.setattr(ingest, "DOCUMENT_CACHE_PATH", cache_path)
    monkeypatch.setattr(ingest, "scrape_blog_documents", lambda: blog_documents)

    documents = ingest.load_blog_documents(refresh_cache=True)

    assert documents == blog_documents
    assert ingest._read_document_cache(cache_path) == blog_documents


def test_ingest_passes_refresh_cache_to_document_loader(monkeypatch) -> None:
    calls: list[bool] = []

    def fake_load_blog_documents(refresh_cache: bool = False) -> list[Document]:
        calls.append(refresh_cache)
        return [
            Document(
                page_content="React performance guidance.",
                metadata={
                    "source": "https://www.bitovi.com/blog/react-performance",
                    "title": "React Performance",
                },
            )
        ]

    class FakeSplitter:
        @classmethod
        def from_tiktoken_encoder(cls, **_: object) -> "FakeSplitter":
            return cls()

        def split_documents(self, documents: list[Document]) -> list[Document]:
            return documents

    class FakeCollection:
        def __init__(self) -> None:
            self.count_value = 0

        def count(self) -> int:
            return self.count_value

    class FakeChroma:
        def __init__(self, **_: object) -> None:
            self._collection = FakeCollection()

        def add_documents(self, documents: list[Document]) -> None:
            self._collection.count_value += len(documents)

    monkeypatch.setattr(ingest, "load_blog_documents", fake_load_blog_documents)
    monkeypatch.setattr(ingest, "RecursiveCharacterTextSplitter", FakeSplitter)
    monkeypatch.setattr(ingest, "Chroma", FakeChroma)
    monkeypatch.setattr(ingest, "get_embeddings", lambda: object())

    assert ingest.ingest(refresh_cache=True) == (1, 1)
    assert calls == [True]
