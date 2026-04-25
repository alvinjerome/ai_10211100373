# Architecture

## Data Flow
User Query -> Query Expansion (light rewrite, optional) -> Hybrid Retrieval (vector + keyword) -> Context Selection -> Prompt Construction -> LLM -> Final Response

## Components
1. `build_index.py`
   - Downloads dataset sources
   - Loads CSV/PDF
   - Applies manual chunking
   - Builds embeddings and FAISS index
2. `src/retriever.py`
   - Embedding-based semantic retrieval
   - TF-IDF keyword retrieval
   - Hybrid score fusion
3. `src/pipeline.py`
   - Prompt engineering
   - Context window truncation
   - LLM call + fallback mode
   - Run logging
4. `app.py`
   - UI for querying
   - Displays retrieved chunks, scores, and final prompt
5. `evaluate.py`
   - Adversarial queries
   - RAG vs baseline comparison
   - Hallucination/consistency proxy metrics

## Why this design fits the domain
- Budget and election data need grounded answers.
- Hybrid retrieval reduces irrelevant chunk issues from either pure semantic or pure keyword search.
- Prompt explicitly constrains model behavior and asks for evidence citation.
