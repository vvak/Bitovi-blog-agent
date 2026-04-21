from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

os.environ.setdefault("USER_AGENT", "bitovi-blog-rag/0.1")

from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_community.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import BLOG_URL, CHROMA_DIR, COLLECTION_NAME, get_embeddings

ADD_BATCH_SIZE = 64
DOCUMENT_CACHE_PATH = Path(".cache/bitovi_blog_documents.json")


def _clean_lines(text: str) -> list[str]:
    """Strip blank lines and known Bitovi page chrome from extracted text."""
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("<img ") or line in {'">', "|"}:
            continue
        if "facebook.com/tr?" in line:
            continue
        if line.startswith("ServicesProject Management"):
            continue
        if line.startswith("Contact Us(312)"):
            continue
        if line.startswith("Contact Bitovi(312)"):
            continue
        if line.startswith("Bitovi Blog - UX and UI design"):
            continue
        lines.append(line)
    return lines


def _metadata_from_content(content: str, fallback_url: str) -> dict[str, str]:
    """Extract source, title, and publish date metadata from a crawled page."""
    soup = BeautifulSoup(content, "html.parser")
    lines = _clean_lines(soup.get_text("\n") if "<" in content else content)

    title_tag = (
        soup.find("meta", property="og:title")
        or soup.find("meta", attrs={"name": "twitter:title"})
        or soup.find("title")
    )
    date_tag = (
        soup.find("meta", property="article:published_time")
        or soup.find("meta", attrs={"name": "publish_date"})
        or soup.find("time")
    )

    title = ""
    if title_tag:
        title = title_tag.get("content") or title_tag.get_text(" ", strip=True)
    if not title and lines:
        title = lines[0]

    publish_date = ""
    if date_tag:
        publish_date = (
            date_tag.get("datetime")
            or date_tag.get("content")
            or date_tag.get_text(" ", strip=True)
        )

    return {
        "source": fallback_url,
        "title": title.strip() or fallback_url,
        "publish_date": publish_date.strip(),
    }


def _extract_article_text(content: str, title: str) -> str:
    """Keep the main article body and trim repeated sharing/navigation content."""
    soup = BeautifulSoup(content, "html.parser")
    article = soup.select_one("article.blog-article") or soup.find("article")
    if article:
        lines = _clean_lines(article.get_text("\n", strip=True))
    else:
        lines = _clean_lines(content)

    start_index = 0
    if "Share:" in lines:
        start_index = lines.index("Share:") + 1

    end_index = len(lines)
    for marker in ("Previous Post", "Next Post", "Contact Bitovi(312)"):
        if marker in lines[start_index:]:
            end_index = min(end_index, lines.index(marker, start_index))

    article_lines = lines[start_index:end_index]
    if title and (not article_lines or article_lines[0] != title):
        article_lines.insert(0, title)

    return "\n".join(article_lines).strip()


def _normalize_metadata(documents: Iterable[Document]) -> list[Document]:
    """Convert raw loader documents into article-only documents with stable metadata."""
    normalized = []
    for doc in documents:
        source = doc.metadata.get("source") or doc.metadata.get("loc") or BLOG_URL
        metadata = _metadata_from_content(doc.page_content, source)
        metadata.update({k: v for k, v in doc.metadata.items() if isinstance(v, str)})
        metadata["source"] = source
        metadata["title"] = metadata.get("title") or source
        metadata["publish_date"] = metadata.get("publish_date") or ""
        page_content = _extract_article_text(doc.page_content, metadata["title"])
        if page_content:
            normalized.append(Document(page_content=page_content, metadata=metadata))
    return normalized


def _count_sources(documents: Iterable[Document]) -> int:
    """Count unique source URLs represented by a document collection."""
    return len(
        {
            doc.metadata.get("source") or doc.metadata.get("loc")
            for doc in documents
            if doc.metadata.get("source") or doc.metadata.get("loc")
        }
    )


def _load_with_sitemap() -> list[Document]:
    """Load Bitovi blog pages listed in the public sitemap."""
    sitemap_url = "https://www.bitovi.com/sitemap.xml"
    loader = SitemapLoader(
        web_path=sitemap_url,
        filter_urls=[r"https://www\.bitovi\.com/blog/.+"],
    )
    return loader.load()


def _load_with_recursive_url() -> list[Document]:
    """Crawl the blog recursively when sitemap loading is unavailable."""
    loader = RecursiveUrlLoader(
        url=BLOG_URL,
        max_depth=3,
        prevent_outside=True,
        link_regex=re.compile(r"^https://www\.bitovi\.com/blog/.+"),
    )
    return loader.load()


def _document_to_cache_item(document: Document) -> dict[str, object]:
    """Serialize a LangChain document into the JSON cache shape."""
    return {
        "page_content": document.page_content,
        "metadata": document.metadata,
    }


def _document_from_cache_item(item: dict[str, object]) -> Document:
    """Rebuild a LangChain document from one JSON cache entry."""
    page_content = item.get("page_content")
    metadata = item.get("metadata", {})
    if not isinstance(page_content, str) or not isinstance(metadata, dict):
        raise ValueError("Cached document has invalid shape.")

    return Document(
        page_content=page_content,
        metadata={str(key): value for key, value in metadata.items()},
    )


def _read_document_cache(cache_path: Path = DOCUMENT_CACHE_PATH) -> list[Document]:
    """Load cached blog documents and validate that the cache is usable."""
    with cache_path.open() as cache_file:
        cached_items = json.load(cache_file)
    if not isinstance(cached_items, list):
        raise ValueError("Document cache must contain a JSON array.")

    documents = [_document_from_cache_item(item) for item in cached_items]
    if not documents:
        raise ValueError("Document cache is empty.")

    print(
        f"Loaded {len(documents)} documents from {cache_path} "
        f"across {_count_sources(documents)} source URLs."
    )
    return documents


def _write_document_cache(
    documents: list[Document],
    cache_path: Path = DOCUMENT_CACHE_PATH,
) -> None:
    """Persist normalized blog documents so later ingests can skip scraping."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    with temp_path.open("w") as cache_file:
        json.dump(
            [_document_to_cache_item(document) for document in documents],
            cache_file,
            indent=2,
        )
        cache_file.write("\n")
    temp_path.replace(cache_path)
    print(f"Cached {len(documents)} documents in {cache_path}.")


def scrape_blog_documents() -> list[Document]:
    """Scrape Bitovi blog pages and normalize them into clean article documents."""
    try:
        documents = _load_with_sitemap()
        if not documents:
            raise RuntimeError("Sitemap contained no matching blog URLs.")
    except Exception as exc:
        print(f"Sitemap load failed, falling back to recursive crawl: {exc}")
        documents = _load_with_recursive_url()
        if not documents:
            raise RuntimeError(
                f"No documents loaded from sitemap or recursive crawl for {BLOG_URL}"
            ) from exc

    documents = _normalize_metadata(documents)
    print(f"Loaded {len(documents)} documents from {_count_sources(documents)} source URLs.")
    print(
        f"Extracted text from {len(documents)} documents "
        f"across {_count_sources(documents)} source URLs."
    )
    return documents


def load_blog_documents(refresh_cache: bool = False) -> list[Document]:
    """Load blog documents from cache unless a fresh scrape is requested or required."""
    if not refresh_cache and DOCUMENT_CACHE_PATH.exists():
        try:
            return _read_document_cache(DOCUMENT_CACHE_PATH)
        except Exception as exc:
            print(f"Document cache load failed, scraping blog instead: {exc}")

    documents = scrape_blog_documents()
    _write_document_cache(documents, DOCUMENT_CACHE_PATH)
    return documents


def _batched(documents: list[Document], batch_size: int) -> Iterable[list[Document]]:
    """Yield fixed-size document batches for Chroma insertion."""
    for index in range(0, len(documents), batch_size):
        yield documents[index : index + batch_size]


def ingest(reset: bool = False, refresh_cache: bool = False) -> tuple[int, int]:
    """Build or rebuild the Chroma index from scraped Bitovi blog articles."""
    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    documents = load_blog_documents(refresh_cache=refresh_cache)
    if not documents:
        raise RuntimeError(f"No documents loaded from {BLOG_URL}")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    if not chunks:
        raise RuntimeError(f"No chunks created from {BLOG_URL}")

    source_count = _count_sources(chunks)
    print(f"Created {len(chunks)} chunks from {source_count} source URLs.")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    total_batches = (len(chunks) + ADD_BATCH_SIZE - 1) // ADD_BATCH_SIZE
    for batch_number, batch in enumerate(_batched(chunks, ADD_BATCH_SIZE), start=1):
        vector_store.add_documents(batch)
        print(
            f"Embedded batch {batch_number}/{total_batches} "
            f"({min(batch_number * ADD_BATCH_SIZE, len(chunks))}/{len(chunks)} chunks)."
        )

    indexed_count = vector_store._collection.count()
    print(f"Vector store now contains {indexed_count} chunks.")
    return indexed_count, source_count


def main() -> None:
    """Parse CLI flags and run the ingestion workflow."""
    parser = argparse.ArgumentParser(description="Ingest Bitovi blog articles into Chroma.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing Chroma index before ingesting.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-scrape the Bitovi blog and replace the cached documents before ingesting.",
    )
    args = parser.parse_args()

    chunk_count, source_count = ingest(
        reset=args.reset,
        refresh_cache=args.refresh_cache,
    )
    print(f"Ingested {chunk_count} chunks from {source_count} sources into {CHROMA_DIR}")


if __name__ == "__main__":
    main()
