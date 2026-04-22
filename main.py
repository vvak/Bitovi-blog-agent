from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import get_embeddings
from chain import answer_question

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize the embedding model during API startup."""
    get_embeddings()
    yield


app = FastAPI(title="Bitovi Blog RAG", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    """Request body for asking the Bitovi blog agent a question."""

    question: str = Field(..., min_length=1)


class Source(BaseModel):
    """Source article citation returned with an answer."""

    title: str
    url: str


class AskResponse(BaseModel):
    """Response body containing the generated answer and its citations."""

    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict[str, str]:
    """Report whether the API process is running."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Validate a question, run the RAG agent, and map failures to API errors."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be blank.")

    try:
        result = answer_question(question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("RAG request failed for question: %s", question)
        raise HTTPException(
            status_code=502,
            detail="RAG request failed. Confirm ANTHROPIC_API_KEY and the Chroma index are available.",
        ) from exc

    return AskResponse(**result)
