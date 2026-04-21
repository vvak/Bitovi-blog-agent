from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.documents import Document


@pytest.fixture
def blog_documents() -> list[Document]:
    return [
        Document(
            page_content="React performance guidance from the Bitovi blog.",
            metadata={
                "source": "https://www.bitovi.com/blog/react-performance",
                "title": "React Performance",
            },
        ),
        Document(
            page_content="Frontend testing guidance from the Bitovi blog.",
            metadata={
                "source": "https://www.bitovi.com/blog/frontend-testing",
                "title": "Frontend Testing",
            },
        ),
    ]


@dataclass
class FakeRetriever:
    documents: list[Document]
    queries: list[str] = field(default_factory=list)

    def invoke(self, query: str) -> list[Document]:
        self.queries.append(query)
        return self.documents


@dataclass
class FakeVectorStore:
    documents: list[Document]
    as_retriever_calls: list[dict[str, Any]] = field(default_factory=list)

    def as_retriever(self, **kwargs: Any) -> FakeRetriever:
        self.as_retriever_calls.append(kwargs)
        return FakeRetriever(self.documents)


@dataclass
class FakeExecutor:
    output: str
    documents: list[Document] | None = None
    intermediate_steps: list[tuple[Any, Any]] | None = None
    inputs: list[dict[str, str]] = field(default_factory=list)

    def invoke(self, inputs: dict[str, str]) -> dict[str, Any]:
        self.inputs.append(inputs)
        steps = self.intermediate_steps
        if steps is None and self.documents is not None:
            steps = [(SimpleNamespace(artifact=self.documents), "retrieved docs")]
        return {
            "output": self.output,
            "intermediate_steps": steps or [],
        }


@pytest.fixture
def fake_retriever(blog_documents: list[Document]) -> FakeRetriever:
    return FakeRetriever(blog_documents)


@pytest.fixture
def fake_vector_store(blog_documents: list[Document]) -> FakeVectorStore:
    return FakeVectorStore(blog_documents)


@pytest.fixture
def fake_executor(blog_documents: list[Document]) -> FakeExecutor:
    return FakeExecutor(
        output="Bitovi recommends measuring React rendering work before optimizing.",
        documents=blog_documents,
    )
