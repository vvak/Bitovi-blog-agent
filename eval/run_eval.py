from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chain import answer_question


DEFAULT_DATASET = Path(__file__).with_name("golden_dataset.json")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "source",
    "sources",
    "the",
    "to",
    "with",
}


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open() as dataset_file:
        data = json.load(dataset_file)
    if not isinstance(data, list):
        raise ValueError("Golden dataset must be a JSON array.")
    return data


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _tokens(text: str) -> set[str]:
    return {
        _stem(token)
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS
    }


def _has_source_coverage(sources: list[dict[str, Any]]) -> bool:
    return any(source.get("url") for source in sources)


def _topic_matches(topic: str, searchable_text: str, sources: list[dict[str, Any]]) -> bool:
    topic_tokens = _tokens(topic)
    if not topic_tokens:
        return True

    if {"cite", "article"} & topic_tokens or {"citation", "citations"} & topic_tokens:
        return _has_source_coverage(sources)

    searchable_tokens = _tokens(searchable_text)
    matched = topic_tokens & searchable_tokens
    required_matches = max(1, min(len(topic_tokens), round(len(topic_tokens) * 0.5)))
    return len(matched) >= required_matches


def score_response(
    answer: str,
    sources: list[dict[str, Any]],
    expected_topics: list[str],
) -> dict[str, Any]:
    if not expected_topics:
        return {"score": 1.0, "matched_topics": [], "missed_topics": []}

    source_text = " ".join(
        f"{source.get('title', '')} {source.get('url', '')}" for source in sources
    )
    searchable_text = f"{answer} {source_text}"
    matched_topics = [
        topic
        for topic in expected_topics
        if _topic_matches(topic, searchable_text, sources)
    ]
    missed_topics = [
        topic for topic in expected_topics if topic not in matched_topics
    ]

    return {
        "score": len(matched_topics) / len(expected_topics),
        "matched_topics": matched_topics,
        "missed_topics": missed_topics,
    }


def run_eval(dataset_path: Path) -> list[dict[str, Any]]:
    results = []
    for item in load_dataset(dataset_path):
        response = answer_question(item["question"])
        expected_topics = item.get("expected_topics", [])
        score = score_response(
            response["answer"],
            response["sources"],
            expected_topics,
        )
        results.append(
            {
                "id": item.get("id"),
                "question": item["question"],
                "score": score["score"],
                "matched_topics": score["matched_topics"],
                "missed_topics": score["missed_topics"],
                "answer": response["answer"],
                "sources": response["sources"],
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run golden dataset evaluation.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the golden dataset JSON file.",
    )
    args = parser.parse_args()

    results = run_eval(args.dataset)
    average = sum(result["score"] for result in results) / len(results)
    print(json.dumps({"average_score": average, "results": results}, indent=2))


if __name__ == "__main__":
    main()
