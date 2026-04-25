from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from src.retriever import HybridRetriever


SYSTEM_PROMPT = """You are a careful assistant for academic QA.
Use only the supplied context to answer.
If context is insufficient, say exactly what is missing.
Do not invent facts, numbers, or names.
"""


def build_prompt(query: str, retrieved: List[dict], max_chars: int = 3200) -> str:
    sections = []
    size = 0
    for i, r in enumerate(retrieved, start=1):
        text = r["chunk"].text.strip()
        block = f"[{i}] Source={r['chunk'].source}\n{text}\n"
        if size + len(block) > max_chars:
            break
        sections.append(block)
        size += len(block)
    context = "\n".join(sections)
    return (
        "Answer the user question using only this retrieved context.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n"
        "Return:\n"
        "1) Direct answer\n"
        "2) Evidence bullets with [chunk_id]\n"
        "3) Confidence (High/Medium/Low)\n"
    )


def generate_with_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    load_dotenv()
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or "No response returned."


def fallback_answer(query: str, retrieved: List[dict]) -> str:
    if not retrieved:
        return "I could not find relevant context in the dataset."
    top = retrieved[0]["chunk"]
    return (
        f"No LLM API key detected, so this is an extractive fallback.\n\n"
        f"Closest chunk source: {top.source}\n"
        f"Possible answer evidence:\n{top.text[:900]}"
    )


def run_query(
    retriever: HybridRetriever,
    query: str,
    top_k: int = 5,
    alpha: float = 0.65,
    model: str = "gpt-4o-mini",
) -> Dict:
    results = retriever.retrieve(query=query, top_k=top_k, alpha=alpha)
    prompt = build_prompt(query=query, retrieved=results)
    try:
        answer = generate_with_openai(prompt=prompt, model=model)
        mode = "rag_llm"
    except Exception:
        answer = fallback_answer(query=query, retrieved=results)
        mode = "rag_fallback"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "mode": mode,
        "answer": answer,
        "prompt": prompt,
        "retrieved": [
            {
                "chunk_id": item["chunk"].chunk_id,
                "source": item["chunk"].source,
                "hybrid_score": item["hybrid_score"],
                "vector_score": item["vector_score"],
                "keyword_score": item["keyword_score"],
                "text": item["chunk"].text,
            }
            for item in results
        ],
    }


def log_run(log_path: Path, payload: Dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(f"{payload}\n")
