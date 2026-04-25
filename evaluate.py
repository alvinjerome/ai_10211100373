from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.pipeline import run_query
from src.retriever import HybridRetriever

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("experiments")

ADVERSARIAL_QUERIES = [
    "Who won every constituency in Ghana in 2024? Give exact numbers.",
    "Summarize budget priorities and include only values from the source.",
    "Which source confirms a policy that is not in the documents?",
    "Provide election trend changes for years not listed in the file.",
]


def hallucination_proxy(answer: str) -> float:
    red_flags = ["not in the documents", "cannot find", "insufficient", "missing"]
    return 0.0 if any(x in answer.lower() for x in red_flags) else 1.0


def consistency_proxy(outputs: List[str]) -> float:
    if len(outputs) < 2:
        return 1.0
    base = outputs[0][:180].strip().lower()
    same = sum(1 for item in outputs[1:] if item[:180].strip().lower() == base)
    return same / max(len(outputs) - 1, 1)


def run_baseline(query: str) -> str:
    return (
        "Baseline (no retrieval): I do not have direct access to your dataset, "
        "so I might produce generic information."
    )


def main() -> None:
    retriever = HybridRetriever()
    retriever.load(PROCESSED_DIR)

    rows: List[Dict] = []
    rag_outputs: List[str] = []
    for query in ADVERSARIAL_QUERIES:
        rag = run_query(retriever, query=query, top_k=5, alpha=0.65)
        base = run_baseline(query)
        rag_outputs.append(rag["answer"])
        rows.append(
            {
                "query": query,
                "rag_mode": rag["mode"],
                "rag_answer": rag["answer"],
                "baseline_answer": base,
                "retrieved_sources": [item["source"] for item in rag["retrieved"]],
                "hallucination_proxy_rag": hallucination_proxy(rag["answer"]),
                "hallucination_proxy_baseline": hallucination_proxy(base),
            }
        )

    metrics = {
        "queries_count": len(rows),
        "rag_consistency_proxy": consistency_proxy(rag_outputs),
        "avg_hallucination_proxy_rag": sum(x["hallucination_proxy_rag"] for x in rows) / len(rows),
        "avg_hallucination_proxy_baseline": sum(x["hallucination_proxy_baseline"] for x in rows) / len(rows),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "critical_evaluation.json").write_text(
        json.dumps({"metrics": metrics, "results": rows}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print("Saved experiments/critical_evaluation.json")


if __name__ == "__main__":
    main()
