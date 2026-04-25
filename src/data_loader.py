from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import requests
from pypdf import PdfReader


@dataclass
class Document:
    source: str
    text: str


def download_file(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    target_path.write_bytes(response.content)


def load_pdf(path: Path) -> List[Document]:
    reader = PdfReader(str(path))
    pages = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(Document(source=f"{path.name}:page_{idx + 1}", text=text))
    return pages


def load_csv(path: Path) -> List[Document]:
    docs: List[Document] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            docs.append(Document(source=f"{path.name}:row_{i + 1}", text=str(row)))
    return docs


def load_documents(raw_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for path in raw_dir.glob("*"):
        if path.suffix.lower() == ".pdf":
            docs.extend(load_pdf(path))
        elif path.suffix.lower() == ".csv":
            docs.extend(load_csv(path))
    return docs
