from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = "claude-haiku-4-5"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BLOG_URL = "https://www.bitovi.com/blog"
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "bitovi_blog"
RETRIEVER_K = 3

def get_llm():
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=LLM_MODEL, temperature=0, max_tokens=1024)


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={
        "trust_remote_code": True,
        "device": "mps",         # use Apple Silicon GPU
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64,
            "prompt_name": "document",  # prefix for ingest

        },
    )
