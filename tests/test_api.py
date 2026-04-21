from __future__ import annotations

from fastapi.testclient import TestClient

import main


def test_health_endpoint() -> None:
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_endpoint_returns_answer_and_sources(monkeypatch) -> None:
    def fake_answer_question(question: str) -> dict[str, object]:
        assert question == "What does Bitovi say about testing?"
        return {
            "answer": "Bitovi recommends focused automated tests.",
            "sources": [
                {
                    "title": "Frontend Testing",
                    "url": "https://www.bitovi.com/blog/frontend-testing",
                }
            ],
        }

    monkeypatch.setattr(main, "answer_question", fake_answer_question)
    client = TestClient(main.app)

    response = client.post(
        "/ask",
        json={"question": "  What does Bitovi say about testing?  "},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Bitovi recommends focused automated tests.",
        "sources": [
            {
                "title": "Frontend Testing",
                "url": "https://www.bitovi.com/blog/frontend-testing",
            }
        ],
    }


def test_ask_endpoint_rejects_blank_question() -> None:
    client = TestClient(main.app)

    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422
    assert response.json()["detail"] == "Question cannot be blank."


def test_ask_endpoint_maps_missing_vector_store_to_503(monkeypatch) -> None:
    def fake_answer_question(_: str) -> dict[str, object]:
        raise RuntimeError("Vector store not found at chroma_db.")

    monkeypatch.setattr(main, "answer_question", fake_answer_question)
    client = TestClient(main.app)

    response = client.post("/ask", json={"question": "What does Bitovi say?"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Vector store not found at chroma_db."


def test_ask_endpoint_maps_unexpected_agent_failure_to_502(monkeypatch) -> None:
    def fake_answer_question(_: str) -> dict[str, object]:
        raise ValueError("Anthropic request failed")

    monkeypatch.setattr(main, "answer_question", fake_answer_question)
    client = TestClient(main.app)

    response = client.post("/ask", json={"question": "What does Bitovi say?"})

    assert response.status_code == 502
    assert response.json() == {
        "detail": "RAG request failed. Confirm ANTHROPIC_API_KEY and the Chroma index are available."
    }
