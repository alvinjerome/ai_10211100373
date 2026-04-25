# AI Exam RAG Project (Manual Pipeline)

This project implements a full RAG assistant without LangChain/LlamaIndex, following the exam requirements:
- Manual data cleaning/loading
- Manual chunking strategy
- Manual embedding + vector storage
- Hybrid retrieval (keyword + vector)
- Prompt engineering + context window control
- Critical evaluation with adversarial queries
- UI showing retrieved chunks, similarity scores, and final prompt

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI key in `.env`:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

## 2) Build the RAG Index

```bash
python build_index.py
```

This downloads:
- Ghana election CSV
- 2025 budget PDF

And builds:
- `data/processed/faiss.index`
- `data/processed/embeddings.npy`
- `data/processed/chunks.json`

## 3) Run the App

```bash
streamlit run app.py
```

## 4) Run Evaluation

```bash
python evaluate.py
```

Output:
- `experiments/critical_evaluation.json`

## 5) Rubric Mapping

- **Part A (Data Engineering):** `src/data_loader.py`, `src/chunking.py`
- **Part B (Custom Retrieval):** `src/retriever.py`
- **Part C (Critical Evaluation):** `evaluate.py`, `experiments/`
- **Part D (Prompt Engineering):** `src/pipeline.py`
- **Part E (Full Pipeline):** `app.py`, `src/pipeline.py`
- **Part F (Architecture):** `docs/architecture.md`
- **Part G (Innovation):** hybrid retrieval scoring with adjustable alpha in UI

## 6) Required Screenshot Checklist (for report)

Take screenshots of:
1. App main page with query input and retrieval settings.
2. Final response section after running a query.
3. Retrieved chunks panel showing:
   - source
   - hybrid/vector/keyword scores
4. Prompt sent to LLM section.
5. Terminal output for:
   - `python build_index.py`
   - `python evaluate.py`
6. `experiments/critical_evaluation.json` opened in editor.
7. One completed manual log in `experiments/manual_experiment_log_template.md`.
8. Architecture file `docs/architecture.md`.

## 7) Deployment (Fastest: Streamlit Community Cloud)

1. Push this project to GitHub.
2. Go to [https://share.streamlit.io](https://share.streamlit.io).
3. Click **New app** and select your repo + branch.
4. Set main file to `app.py`.
5. Add secrets in Streamlit dashboard:
   - `OPENAI_API_KEY`
   - `OPENAI_MODEL`
6. Deploy.

If the app starts without index files, open Streamlit app console and run:
- `python build_index.py`

(Alternative: commit `data/processed/*` if allowed by your instructor.)
