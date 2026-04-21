from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document

import chain


def test_dedupe_sources_prefers_source_and_preserves_first_title() -> None:
    documents = [
        Document(
            page_content="first",
            metadata={
                "source": "https://www.bitovi.com/blog/react-performance",
                "title": "React Performance",
            },
        ),
        Document(
            page_content="duplicate",
            metadata={
                "source": "https://www.bitovi.com/blog/react-performance",
                "title": "Duplicate Title",
            },
        ),
        Document(
            page_content="fallback url",
            metadata={"url": "https://www.bitovi.com/blog/testing"},
        ),
        Document(page_content="missing source", metadata={"title": "No URL"}),
    ]

    assert chain._dedupe_sources(documents) == [
        {
            "title": "React Performance",
            "url": "https://www.bitovi.com/blog/react-performance",
        },
        {
            "title": "https://www.bitovi.com/blog/testing",
            "url": "https://www.bitovi.com/blog/testing",
        },
    ]


def test_docs_from_steps_extracts_documents_from_supported_agent_outputs() -> None:
    artifact_doc = Document(page_content="artifact", metadata={"source": "artifact"})
    observation_doc = Document(page_content="observation", metadata={"source": "observation"})
    tuple_doc = Document(page_content="tuple", metadata={"source": "tuple"})
    ignored_doc = "not a document"

    steps = [
        (SimpleNamespace(artifact=[artifact_doc, ignored_doc]), "text observation"),
        (SimpleNamespace(), [observation_doc, ignored_doc]),
        (SimpleNamespace(), ("content", [tuple_doc, ignored_doc])),
    ]

    assert chain._docs_from_steps(steps) == [artifact_doc, observation_doc, tuple_doc]


def test_get_retriever_configures_similarity_search(monkeypatch, fake_vector_store) -> None:
    monkeypatch.setattr(chain, "get_vector_store", lambda: fake_vector_store)

    retriever = chain.get_retriever()

    assert retriever.documents
    assert fake_vector_store.as_retriever_calls == [
        {
            "search_type": "similarity",
            "search_kwargs": {"k": chain.RETRIEVER_K},
        }
    ]
