from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.pipeline import log_run, run_query
from src.retriever import HybridRetriever

PROCESSED_DIR = Path("data/processed")
LOG_PATH = Path("logs/query_runs.log")

st.set_page_config(page_title="Exam RAG Assistant", layout="wide")
st.title("RAG Assistant - AI Exam Project")
st.caption("Manual pipeline: chunking, embeddings, hybrid retrieval, prompt construction, LLM response.")


@st.cache_resource(show_spinner=False)
def load_retriever() -> HybridRetriever:
    retriever = HybridRetriever()
    retriever.load(PROCESSED_DIR)
    return retriever


if not PROCESSED_DIR.exists():
    st.error("Index not found. Run: `python build_index.py` first.")
    st.stop()

retriever = load_retriever()

with st.sidebar:
    st.header("Retrieval Settings")
    top_k = st.slider("Top-K Chunks", min_value=2, max_value=10, value=5, step=1)
    alpha = st.slider("Hybrid Weight (vector alpha)", min_value=0.1, max_value=0.9, value=0.65, step=0.05)
    model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    st.info("If OPENAI_API_KEY is missing, the app uses an extractive fallback answer.")

query = st.text_area("Ask a question about the dataset", height=110, placeholder="Type your question...")

if st.button("Run RAG Query", type="primary") and query.strip():
    payload = run_query(retriever=retriever, query=query.strip(), top_k=top_k, alpha=alpha, model=model)
    log_run(LOG_PATH, payload)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Final Response")
        st.write(payload["answer"])
    with col2:
        st.subheader("Run Metadata")
        st.json(
            {
                "timestamp": payload["timestamp"],
                "mode": payload["mode"],
                "top_k": top_k,
                "alpha": alpha,
                "retrieved_count": len(payload["retrieved"]),
            }
        )

    st.subheader("Retrieved Chunks + Similarity Scores")
    for item in payload["retrieved"]:
        with st.expander(
            f"chunk_id={item['chunk_id']} | source={item['source']} | hybrid={item['hybrid_score']:.4f}"
        ):
            st.write(
                {
                    "vector_score": round(item["vector_score"], 4),
                    "keyword_score": round(item["keyword_score"], 4),
                }
            )
            st.write(item["text"])

    st.subheader("Prompt Sent To LLM")
    st.code(payload["prompt"])
