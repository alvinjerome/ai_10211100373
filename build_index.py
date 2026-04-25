from __future__ import annotations

from pathlib import Path

from src.chunking import build_chunks
from src.data_loader import download_file, load_documents
from src.retriever import HybridRetriever

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

DATA_SOURCES = [
    (
        "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv",
        RAW_DIR / "ghana_election_result.csv",
    ),
    (
        "https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy.pdf",
        RAW_DIR / "2025_budget_statement.pdf",
    ),
]


def main() -> None:
    for url, target in DATA_SOURCES:
        if not target.exists():
            print(f"Downloading: {url}")
            download_file(url, target)

    docs = load_documents(RAW_DIR)
    print(f"Loaded {len(docs)} raw document units")
    chunks = build_chunks(docs, chunk_size=700, overlap=140)
    print(f"Built {len(chunks)} chunks")

    retriever = HybridRetriever()
    retriever.fit(chunks)
    retriever.save(PROCESSED_DIR)
    print(f"Saved index + metadata in {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
