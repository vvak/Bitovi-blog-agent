from __future__ import annotations

from langchain_core.documents import Document

import chain
from tests.conftest import FakeExecutor, FakeRetriever


def test_answer_question_uses_agent_output_and_intermediate_step_sources(
    monkeypatch,
    fake_executor: FakeExecutor,
) -> None:
    def fail_retriever() -> FakeRetriever:
        raise AssertionError("retriever fallback should not be used when agent returns docs")

    monkeypatch.setattr(chain, "get_chain", lambda: fake_executor)
    monkeypatch.setattr(chain, "get_retriever", fail_retriever)

    result = chain.answer_question("How should React performance be handled?")

    assert fake_executor.inputs == [
        {"input": "How should React performance be handled?"}
    ]
    assert result == {
        "answer": "Bitovi recommends measuring React rendering work before optimizing.",
        "sources": [
            {
                "title": "React Performance",
                "url": "https://www.bitovi.com/blog/react-performance",
            },
            {
                "title": "Frontend Testing",
                "url": "https://www.bitovi.com/blog/frontend-testing",
            },
        ],
    }


def test_answer_question_falls_back_to_retriever_when_agent_returns_no_documents(
    monkeypatch,
) -> None:
    fallback_docs = [
        Document(
            page_content="Fallback source",
            metadata={
                "source": "https://www.bitovi.com/blog/fallback",
                "title": "Fallback Article",
            },
        )
    ]
    executor = FakeExecutor(output="Fallback answer.", intermediate_steps=[])
    retriever = FakeRetriever(fallback_docs)

    monkeypatch.setattr(chain, "get_chain", lambda: executor)
    monkeypatch.setattr(chain, "get_retriever", lambda: retriever)

    result = chain.answer_question("What does the blog say?")

    assert retriever.queries == ["What does the blog say?"]
    assert result == {
        "answer": "Fallback answer.",
        "sources": [
            {
                "title": "Fallback Article",
                "url": "https://www.bitovi.com/blog/fallback",
            }
        ],
    }
