from __future__ import annotations

import json
import re
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool

from config import CHROMA_DIR, COLLECTION_NAME, RETRIEVER_K, get_embeddings, get_llm

PROMPT = """Answer questions about Bitovi blog articles.

You have four tools:
- search_blog: semantic search for questions about a specific TOPIC (e.g., "React performance", "design systems").
- get_latest_blog_posts: returns articles sorted by publish date. Use this for ANY question about recency: "latest", "newest", "most recent", "current", "what's new", etc.
- count_articles: counts how many Bitovi blog articles cover a given topic (based on Bitovi's topic tags). Use this for ANY question about how many or how much coverage Bitovi has on a topic (e.g., "how many articles about AI?", "does Bitovi write a lot about testing?").
- list_articles: lists all Bitovi blog articles tagged with a specific topic. Use this for ANY question asking to "show all", "list all", or "get all" articles about a topic (e.g., "Can you show me all articles about DevOps?", "List all React articles"). Input should be a single topic word like "ai", "devops", "react".

If search_blog results are weak, try once more with a better query. If still insufficient, say the Bitovi blog does not contain enough information. Always cite which articles informed the answer. Keep answers concise and factual.

Tools:
{tools}

Format:
Question: input
Thought: reasoning
Action: one of [{tool_names}]
Action Input: query
Observation: result
... repeat as needed
Thought: final
Final Answer: answer

Question: {input}
Thought:{agent_scratchpad}"""

LATEST_POSTS_N = 5
MAX_SOURCES = 5
ARTICLE_LIST_RE = re.compile(
    r"\b(show|list|give|find)\b.*\ball\b.*\b(blog\s+posts?|posts?|articles?)\b"
    r"|\b(which|what)\b.*\ball\b.*\b(blog\s+posts?|posts?|articles?)\b",
    re.IGNORECASE,
)
MONTH_DATE_RE = re.compile(
    r"\b("
    r"January|February|March|April|May|June|July|August|September|October|"
    r"November|December"
    r")\s+\d{1,2},\s+\d{4}\b"
)
PUBLISH_DATE_KEYS = ("publish_date", "published", "date")
DOCUMENT_CACHE_PATH = Path(".cache/bitovi_blog_documents.json")


def _dedupe_sources(documents: list[Document]) -> list[dict[str, str]]:
    """Return unique source citations from retrieved documents in first-seen order."""
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for doc in documents:
        url = doc.metadata.get("source") or doc.metadata.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append(
            {
                "title": doc.metadata.get("title") or url,
                "url": url,
            }
        )
    return sources


def _is_article_list_question(question: str) -> bool:
    """Detect questions that require listing indexed articles."""
    return bool(ARTICLE_LIST_RE.search(question))


def _article_topic(question: str) -> str:
    """Extract the requested article topic, if the question includes one."""
    normalized = question.strip().rstrip("?.!").strip()
    patterns = (
        r"\b(?:about|on|regarding|related to|mention(?:ing)?|discuss(?:ing)?)\s+(.+)$",
        r"\ball\s+(.+?)\s+(?:blog\s+posts?|posts?|articles?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            topic = match.group(1).strip()
            return re.sub(r"^(?:bitovi'?s?\s+)?", "", topic, flags=re.IGNORECASE)
    return ""


def _parse_date(value: Any) -> date | None:
    """Parse supported blog metadata date formats into comparable dates."""
    if not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    iso_match = re.search(r"\d{4}-\d{2}-\d{2}", value)
    if iso_match:
        return date.fromisoformat(iso_match.group(0))

    month_match = MONTH_DATE_RE.search(value)
    if month_match:
        return datetime.strptime(month_match.group(0), "%B %d, %Y").date()

    return None


def _metadata_date(metadata: dict[str, Any], keys: tuple[str, ...]) -> date | None:
    """Return the first parseable date from the requested metadata keys."""
    for key in keys:
        parsed = _parse_date(metadata.get(key))
        if parsed:
            return parsed
    return None


def _format_date(value: date) -> str:
    """Format dates without platform-specific strftime flags."""
    return f"{value.strftime('%B')} {value.day}, {value.year}"


def _all_indexed_documents() -> list[Document]:
    """Load all stored documents with metadata for corpus-wide questions."""
    data = get_document_store().get(include=["documents", "metadatas"])
    contents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    if contents or metadatas:
        return [
            Document(page_content=content or "", metadata=metadata or {})
            for content, metadata in zip(contents, metadatas, strict=False)
        ]

    if DOCUMENT_CACHE_PATH.exists():
        with DOCUMENT_CACHE_PATH.open() as cache_file:
            cached_items = json.load(cache_file)
        if isinstance(cached_items, list):
            return [
                Document(
                    page_content=item.get("page_content") or "",
                    metadata=item.get("metadata") or {},
                )
                for item in cached_items
                if isinstance(item, dict)
            ]

    return []


def _unique_source_documents(documents: list[Document]) -> list[Document]:
    """Collapse chunk documents into one searchable document per source URL."""
    by_source: dict[str, tuple[int, dict[str, Any], list[str]]] = {}
    for order, doc in enumerate(documents):
        source = doc.metadata.get("source") or doc.metadata.get("url")
        if not source:
            continue

        if source not in by_source:
            by_source[source] = (order, dict(doc.metadata), [])
        by_source[source][2].append(doc.page_content)

    return [
        Document(page_content="\n".join(contents), metadata=metadata)
        for _, metadata, contents in sorted(by_source.values(), key=lambda item: item[0])
    ]


def _topic_matchers(topic: str) -> list[re.Pattern[str]]:
    """Build conservative regexes for matching a requested article topic."""
    normalized = topic.strip().lower()
    if normalized in {"ai", "a.i."}:
        return [
            re.compile(r"\bAI\b", re.IGNORECASE),
            re.compile(r"\bartificial intelligence\b", re.IGNORECASE),
        ]

    stripped = topic.strip()
    escaped = re.escape(stripped)
    prefix = r"\b" if stripped[:1].isalnum() else ""
    suffix = r"\b" if stripped[-1:].isalnum() else ""
    return [re.compile(f"{prefix}{escaped}{suffix}", re.IGNORECASE)]


def _metadata_topics(document: Document) -> list[str]:
    """Return normalized article topics stored in metadata."""
    topics = document.metadata.get("topics")
    if not isinstance(topics, str):
        return []
    return [topic.strip().casefold() for topic in topics.split(",") if topic.strip()]


def _matches_topic(document: Document, topic: str) -> bool:
    """Return whether an indexed article matches a requested topic."""
    metadata_topics = _metadata_topics(document)
    if metadata_topics:
        return topic.strip().casefold() in metadata_topics

    haystack = "\n".join(
        [
            document.metadata.get("title", ""),
            document.page_content,
            " ".join(str(value) for value in document.metadata.values()),
        ]
    )
    return any(matcher.search(haystack) for matcher in _topic_matchers(topic))


def _topic_documents(documents: list[Document], topic: str) -> list[Document]:
    """Return documents for a topic, preferring explicit topic metadata."""
    normalized_topic = topic.strip().casefold()
    tagged_documents = [
        document
        for document in documents
        if normalized_topic in _metadata_topics(document)
    ]
    if tagged_documents and normalized_topic not in {"ai", "a.i."}:
        return tagged_documents
    return [document for document in documents if _matches_topic(document, topic)]


def _format_article_list_item(index: int, document: Document) -> str:
    """Format one article for a corpus-wide list answer."""
    title = document.metadata.get("title") or "Untitled Bitovi blog post"
    published = _metadata_date(document.metadata, PUBLISH_DATE_KEYS)
    date_part = f" ({_format_date(published)})" if published else ""
    return f"{index}. {title}{date_part}"


def _answer_article_list(question: str) -> dict[str, Any]:
    """Answer list-all questions by scanning unique indexed source articles."""
    documents = _unique_source_documents(_all_indexed_documents())
    topic = _article_topic(question)
    matches = _topic_documents(documents, topic) if topic else documents
    count = len(matches)
    noun = "article" if count == 1 else "articles"
    if topic:
        answer = f'The indexed Bitovi blog has {count} {noun} about "{topic}":'
    else:
        answer = f"The indexed Bitovi blog has {count} {noun}:"
    if matches:
        answer = "\n".join(
            [answer]
            + [
                _format_article_list_item(index, document)
                for index, document in enumerate(matches, start=1)
            ]
        )
    return {
        "answer": answer,
        "sources": _dedupe_sources(matches),
    }


def _docs_from_steps(steps: list[tuple[Any, Any]]) -> list[Document]:
    """Extract retrieved documents from LangChain agent intermediate steps."""
    docs: list[Document] = []
    for action, observation in steps:
        artifact = getattr(action, "artifact", None)
        if isinstance(artifact, list):
            docs.extend(doc for doc in artifact if isinstance(doc, Document))
        if isinstance(observation, list):
            docs.extend(doc for doc in observation if isinstance(doc, Document))
        elif isinstance(observation, tuple):
            docs.extend(
                doc
                for item in observation
                if isinstance(item, list)
                for doc in item
                if isinstance(doc, Document)
            )
    return docs


def get_retriever():
    """Create a similarity retriever over the persisted Bitovi Chroma collection."""
    return get_vector_store().as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )


def _get_latest_blog_posts_impl(_query: str) -> str:
    """Retrieve the most recent Bitovi blog articles by publish date."""
    vs = get_vector_store()
    result = vs._collection.get(
        include=["metadatas", "documents"],
        where={"publish_timestamp": {"$gt": 0}},
    )

    metadatas: list[dict] = result.get("metadatas") or []
    documents: list[str] = result.get("documents") or []

    paired = sorted(
        zip(metadatas, documents),
        key=lambda pair: pair[0].get("publish_timestamp", 0),
        reverse=True,
    )

    seen_sources: set[str] = set()
    articles: list[dict] = []
    for meta, content in paired:
        source = meta.get("source", "")
        if not source or source in seen_sources:
            continue
        seen_sources.add(source)
        articles.append(
            {
                "title": meta.get("title", source),
                "url": source,
                "publish_date": meta.get("publish_date") or meta.get("lastmod") or "unknown",
                "snippet": content[:300].replace("\n", " "),
            }
        )
        if len(articles) >= LATEST_POSTS_N:
            break

    if not articles:
        return "No articles with known publication dates found in the index."

    lines = ["Here are the most recent Bitovi blog posts:\n"]
    for i, art in enumerate(articles, 1):
        lines.append(
            f"{i}. [{art['title']}]({art['url']})\n"
            f"   Published: {art['publish_date']}\n"
            f"   Snippet: {art['snippet']}\n"
        )
    return "\n".join(lines)


def _count_articles_impl(query: str) -> str:
    """Count indexed Bitovi blog articles tagged with a matching topic."""
    slug = query.strip().lower().replace(" ", "-")
    vs = get_vector_store()
    all_meta = vs._collection.get(include=["metadatas"]).get("metadatas", [])
    seen: set[str] = set()
    count = 0
    for meta in all_meta:
        src = meta.get("source", "")
        if not src or src in seen:
            continue
        seen.add(src)
        topics = meta.get("topics", "")
        if slug in [t.strip().lower() for t in topics.split(",") if t.strip()]:
            count += 1
    if count == 0:
        return f"No Bitovi blog articles are tagged with the topic '{query}'."
    return f"The Bitovi blog has {count} articles tagged with the topic '{query}'."


def _list_articles_impl(query: str) -> str:
    """List all Bitovi blog articles tagged with a matching topic."""
    slug = query.strip().lower().replace(" ", "-")
    vs = get_vector_store()
    all_meta = vs._collection.get(include=["metadatas"]).get("metadatas", [])

    seen: set[str] = set()
    articles: list[dict[str, str]] = []

    for meta in all_meta:
        src = meta.get("source", "")
        if not src or src in seen:
            continue
        seen.add(src)
        topics = meta.get("topics", "")
        if slug in [t.strip().lower() for t in topics.split(",") if t.strip()]:
            articles.append(
                {
                    "title": meta.get("title", src),
                    "url": src,
                    "publish_date": meta.get("publish_date")
                    or meta.get("lastmod")
                    or "unknown",
                }
            )

    if not articles:
        return f"No Bitovi blog articles are tagged with the topic '{query}'."

    lines = [f"Here are all {len(articles)} Bitovi blog articles about {query}:\n"]
    for i, art in enumerate(articles, 1):
        lines.append(
            f"{i}. [{art['title']}]({art['url']})\n"
            f"   Published: {art['publish_date']}\n"
        )
    return "\n".join(lines)


@lru_cache(maxsize=1)
def get_document_store() -> Chroma:
    """Open Chroma for metadata/document reads without loading embeddings."""
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"Vector store not found at {CHROMA_DIR}. Run `python ingest.py` first."
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Open the persisted Chroma vector store, failing clearly if it is missing."""
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"Vector store not found at {CHROMA_DIR}. Run `python ingest.py` first."
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


@lru_cache(maxsize=1)
def get_chain():
    """Build the cached ReAct agent executor with Bitovi blog search and recency tools."""
    retriever = get_retriever()
    rag_tool = create_retriever_tool(
        retriever=retriever,
        name="search_blog",
        description="Search the Bitovi blog for information about a specific topic. Use this tool whenever the question is about Bitovi content. Input should be a search query string.",
        response_format="content_and_artifact",
    )
    latest_tool = Tool(
        name="get_latest_blog_posts",
        func=_get_latest_blog_posts_impl,
        description=(
            "Return the most recent Bitovi blog articles sorted by publication date. "
            "Use this tool when the question asks about the latest, newest, most recent, "
            "or current blog posts. Do NOT use search_blog for recency questions. "
            "Input can be any string (it is ignored)."
        ),
    )
    count_tool = Tool(
        name="count_articles",
        func=_count_articles_impl,
        description=(
            "Count Bitovi blog articles by topic tag. "
            "Use this when asked 'how many articles', 'how much coverage', or similar counting questions. "
            "Input should be a single topic word matching a Bitovi blog topic (e.g., 'ai', 'react', 'testing'). "
            "Returns the exact count from Bitovi's topic pages."
        ),
    )
    list_tool = Tool(
        name="list_articles",
        func=_list_articles_impl,
        description=(
            "List all Bitovi blog articles tagged with a specific topic. "
            "Use this when asked to 'show all', 'list all', or 'get all' articles about a topic. "
            "Input should be a single topic word matching a Bitovi blog topic (e.g., 'ai', 'devops', 'react'). "
            "Returns a formatted list with titles, URLs, and publish dates."
        ),
    )
    tools = [rag_tool, latest_tool, count_tool, list_tool]
    agent = create_react_agent(get_llm(), tools=tools, prompt=PromptTemplate.from_template(PROMPT))
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=4,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


def _extract_sources_from_text(text: str) -> list[dict[str, str]]:
    """Extract markdown links [title](url) from formatted text."""
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in re.finditer(r'\[([^\]]+)\]\((https://[^\)]+)\)', text):
        title, url = match.groups()
        if url not in seen:
            sources.append({"title": title, "url": url})
            seen.add(url)
    return sources


def _extract_sources_from_tool_output(
    intermediate_steps: list[tuple[Any, Any]],
    answer: str,
) -> list[dict[str, str]]:
    """Extract sources from tool outputs that are actually cited in the answer."""
    tool_sources: list[dict[str, str]] = []
    for action, observation in intermediate_steps:
        if hasattr(action, "tool") and action.tool in {
            "get_latest_blog_posts",
            "list_articles",
        }:
            if isinstance(observation, str):
                tool_sources.extend(_extract_sources_from_text(observation))

    if not tool_sources:
        return []

    # Filter to only sources mentioned in the answer by matching titles
    cited_sources: list[dict[str, str]] = []
    for source in tool_sources:
        # Check if the title or URL appears in the answer
        if source["title"] in answer or source["url"] in answer:
            cited_sources.append(source)

    return cited_sources if cited_sources else tool_sources[:1]


def answer_question(question: str) -> dict[str, Any]:
    """Run the agent for a question and return its answer with source citations."""
    if _is_article_list_question(question):
        return _answer_article_list(question)

    result = get_chain().invoke({"input": question})
    docs = _docs_from_steps(result.get("intermediate_steps", []))

    if not docs:
        # Try to extract sources from tool outputs (for formatted tools like get_latest_blog_posts)
        tool_sources = _extract_sources_from_tool_output(
            result.get("intermediate_steps", []),
            result["output"]
        )
        if tool_sources:
            return {
                "answer": result["output"],
                "sources": tool_sources[:MAX_SOURCES],
            }
        # Fallback to semantic search only if no sources found
        docs = get_retriever().invoke(question)

    return {
        "answer": result["output"],
        "sources": _dedupe_sources(docs)[:MAX_SOURCES],
    }
