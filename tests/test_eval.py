from __future__ import annotations

import importlib.util
from pathlib import Path


def load_run_eval_module():
    module_path = Path(__file__).resolve().parents[1] / "eval" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("run_eval", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_score_response_uses_keyword_overlap_and_sources() -> None:
    run_eval = load_run_eval_module()

    result = run_eval.score_response(
        answer=(
            "Bitovi recommends profiling React rendering work first, then "
            "optimizing the slow parts."
        ),
        sources=[
            {
                "title": "React Performance",
                "url": "https://www.bitovi.com/blog/react-performance",
            }
        ],
        expected_topics=[
            "measure performance before optimizing",
            "rendering behavior",
            "cite Bitovi source articles",
        ],
    )

    assert result["score"] == 1.0
    assert result["missed_topics"] == []


def test_score_response_reports_missed_topics() -> None:
    run_eval = load_run_eval_module()

    result = run_eval.score_response(
        answer="Bitovi discusses React rendering.",
        sources=[],
        expected_topics=[
            "rendering behavior",
            "cite Bitovi source articles",
        ],
    )

    assert result["score"] == 0.5
    assert result["matched_topics"] == ["rendering behavior"]
    assert result["missed_topics"] == ["cite Bitovi source articles"]
