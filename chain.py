from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool

from config import CHROMA_DIR, COLLECTION_NAME, RETRIEVER_K, get_embeddings, get_llm

PROMPT = """Answer questions about Bitovi blog articles.
Use search_blog for any question about Bitovi content. If results are weak, try once more with a better query. If still insufficient, say the Bitovi blog does not contain enough information. Always cite which articles informed the answer. Keep answers concise and factual.

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
    """Build the cached ReAct agent executor with the Bitovi blog search tool."""
    retriever = get_retriever()
    rag_tool = create_retriever_tool(
        retriever=retriever,
        name="search_blog",
        description="Search the Bitovi blog for information. Use this tool whenever the question is about Bitovi content. Input should be a search query string.",
        response_format="content_and_artifact",
    )
    agent = create_react_agent(get_llm(), tools=[rag_tool], prompt=PromptTemplate.from_template(PROMPT))
    return AgentExecutor(
        agent=agent,
        tools=[rag_tool],
        max_iterations=4,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


def answer_question(question: str) -> dict[str, Any]:
    """Run the agent for a question and return its answer with source citations."""
    result = get_chain().invoke({"input": question})
    docs = _docs_from_steps(result.get("intermediate_steps", []))
    if not docs:
        docs = get_retriever().invoke(question)
    return {
        "answer": result["output"],
        "sources": _dedupe_sources(docs),
    }
