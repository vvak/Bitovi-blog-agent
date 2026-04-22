from __future__ import annotations

from langchain_core.documents import Document

import chain
from tests.conftest import FakeExecutor, FakeRetriever, FakeVectorStore


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


def test_answer_question_lists_all_articles_by_topic_metadata(monkeypatch) -> None:
    devops_docs = [
        Document(
            page_content=f"DevOps guidance {index}.",
            metadata={
                "source": f"https://www.bitovi.com/blog/devops-{index}",
                "title": f"DevOps {index}",
                "publish_date": f"2026-03-{index:02d}",
                "topics": "DevOps",
            },
        )
        for index in range(1, 7)
    ]
    duplicate_devops_chunk = Document(
        page_content="More deployment details.",
        metadata={
            "source": "https://www.bitovi.com/blog/devops-1",
            "title": "DevOps 1",
            "publish_date": "2026-03-01",
            "topics": "DevOps",
        },
    )
    incidental_devops_doc = Document(
        page_content="This article mentions DevOps but is about planning.",
        metadata={
            "source": "https://www.bitovi.com/blog/planning",
            "title": "Planning",
            "topics": "Project Management",
        },
    )

    def fail_chain() -> FakeExecutor:
        raise AssertionError("article list questions should not use the agent")

    monkeypatch.setattr(chain, "get_chain", fail_chain)
    monkeypatch.setattr(
        chain,
        "get_document_store",
        lambda: FakeVectorStore(
            [*devops_docs, duplicate_devops_chunk, incidental_devops_doc]
        ),
    )

    result = chain.answer_question("Can you show me all Bitovi articles about DevOps?")

    assert result["answer"] == (
        'The indexed Bitovi blog has 6 articles about "DevOps":\n'
        "1. DevOps 1 (March 1, 2026)\n"
        "2. DevOps 2 (March 2, 2026)\n"
        "3. DevOps 3 (March 3, 2026)\n"
        "4. DevOps 4 (March 4, 2026)\n"
        "5. DevOps 5 (March 5, 2026)\n"
        "6. DevOps 6 (March 6, 2026)"
    )
    assert result["sources"] == [
        {
            "title": f"DevOps {index}",
            "url": f"https://www.bitovi.com/blog/devops-{index}",
        }
        for index in range(1, 7)
    ]
