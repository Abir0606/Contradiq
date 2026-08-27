# ContractIQ — Agentic RAG for Legal Contracts

Agentic RAG system for the **CUAD v1** corpus (510 commercial contracts, 41 clause categories). Demonstrates production-grade **Pinecone** depth and **LangChain/LangGraph** agent design — from naive baseline to reranking + parent-document retrieval + corrective retrieval.

> **Stack**: Pinecone serverless (`dotproduct` 384d) · LangChain · LangGraph · `all-MiniLM-L6-v2` (local, free) + Groq `openai/gpt-oss-120b` (free tier) · `cross-encoder/ms-marco-MiniLM-L-6-v2` · FastAPI · Streamlit · Docker Compose

![Demo placeholder — add screen recording here](docs/demo.gif)

## Architecture

```
                ┌─────────────────────────────────────────────────┐
     TXT (510) ─┤  Ingest: loader ─► chunker (1200/150) ─► MPNet │
CUAD CSV ───────┤         │  BM25 (stemmed, unit-max) ─► Pinecone│
                └─────────┼─────────────────────────────────────┘
                          │ namespaces: baseline | hybrid-v1 | advanced-v1 (child 400/50 + parent_text)
                          ▼
                  ┌───────────────┐      ┌──────────────────────┐
     query ──────►│  Hybrid Retr. │─20──►│ CrossEncoder Rerank  │─6─┐
                  │  (dense+BM25) │      │ parent dedup        │   │
                  └───────────────┘      └──────────────────────┘   │
                          │                                         ▼
                  ┌───────▼────────────────────────────────────────────┐
                  │ LangGraph Agent                                    │
                  │ router (general vs contract_qa) → retrieve → grade │
                  │ → rewrite (×2) → generate → verify (cite check)   │
                  └───────┬────────────────────────────────────────────┘
                          ▼
                     FastAPI (/query, /agent/query, /query/stream, /health)
                          ▼
                     Streamlit (mode toggle, filters, trace, sources)
```

## Stages

| Stage | Status | What |
|-------|--------|------|
| 0 Scaffold | ✅ | `src/contractiq` package, `pydantic-settings` config, `ruff`/`pytest` |
| 1 Ingestion + naive RAG | ✅ | 8,983 chunks, `baseline` namespace, naive chain with citations |
| 2 Pinecone depth | ✅ | `hybrid-v1` dense+BM25 (stemmed), metadata filters (`contract_type`, `part`, `clause_categories`), namespaces |
| 3 LangGraph agent | ✅ | Router → grade → corrective rewrite → citation verifier |
| 4 Reranking + parent | ✅ | `cross-encoder/ms-marco-MiniLM` rerank + parent-document (1200→400) in `advanced-v1` (7,520 children for 150-contract demo; 24,083 full) |
| 5 Evaluation harness | ✅ | 60 golden QAs (20 entity/20 yes/20 no) from CUAD, `scripts/eval.py` with LLM judge + retrieval hit |
| 6 Demo packaging | ✅ | FastAPI + Streamlit + `docker-compose.yml` (local only) |

## Quick Start (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
# .env needs PINECONE_API_KEY, GROQ_API_KEY (free at console.groq.com)

# Ingest (baseline + hybrid already done; advanced demo subset)
.venv\Scripts\python.exe scripts\ingest.py --advanced --limit 150 --reset  # full: omit --limit (~75 min, 24k vectors, no LLM tokens)

# Query
.venv\Scripts\python.exe scripts\query.py --mode hybrid "What is the agreement date for Cybergy Holdings affiliate agreement?"
.venv\Scripts\python.exe scripts\query.py --mode advanced --rerank --parent "termination for convenience?"

# Agent (with trace)
.venv\Scripts\python.exe scripts\agent.py --advanced "what's the deal with ending the Netgear arrangement early?"

# Eval (retrieval-only avoids Groq TPD; full needs quota)
.venv\Scripts\python.exe scripts\build_golden.py --n 60
.venv\Scripts\python.exe scripts\eval.py --modes naive hybrid advanced --retrieval-only --limit 20
.venv\Scripts\python.exe scripts\eval.py --modes naive hybrid advanced --limit 20  # needs Groq quota
```

## API

```powershell
.venv\Scripts\python.exe -m uvicorn contractiq.api.main:app --host 127.0.0.1 --port 8000  # then open http://127.0.0.1:8000/docs
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d '{"question":"Which law governs the Cybergy agreement?","mode":"hybrid"}'
curl http://127.0.0.1:8000/health
```

Streamlit (needs API running):
```powershell
.venv\Scripts\python.exe -m streamlit run src/contractiq/ui/app.py
# or via Docker:
docker compose up --build  # api:8000, ui:8501
```

## Evaluation (12-QA retrieval-only slice, `data/eval/eval_results.json`)

| mode | n | accuracy (=hit) | hit_rate | entity | yes | no | latency |
|------|---|-----------------|----------|--------|-----|----|---------|
| naive | 12 | 0.583 | 0.583 | 0.800 | 1.000 | 0.000 | 8.2s |
| hybrid | 12 | 0.583 | 0.583 | 0.800 | 1.000 | 0.000 | 7.3s |
| advanced | 8 | 0.500 | 0.500 | 0.500 | 1.000 | 0.000 | 11.0s |

Filtered retrieval (`contract_name $in`) — naive/hybrid tie on simple lookups; advanced (parent+rerank) wins on open-domain queries requiring context expansion (see Netgear §15 case in Stage 3). Full 60-QA generation eval: `scripts/eval.py --modes naive hybrid advanced` after Groq 200k TPD reset.

## Key Decisions

- **TXT not PDF** (CUAD TXT avoids OCR), section-aware chunking + flat-doc normalization for 89KB single-line contracts.
- **Dotproduct + normalized MiniLM** for native Pinecone hybrid (cosine index rejected sparse).
- **Hand-rolled BM25** (suffix stemming, unit-max scaling) — `pinecone-text` needs `mmh3` C++ build on Windows.
- **Token budgeting** for Groq free tier (8k TPM): clipped contexts, `max_retries=4`.

## Layout

```
src/contractiq/{config,ingest/{loader,chunker,sparse,vectorstore},retrieval/{factory,naive,hybrid,filtering,reranker,advanced},agent/{state,nodes,graph},api/{main,schemas},ui/app.py}
scripts/{ingest,query,agent,build_golden,eval}.py
data/eval/{golden.json,eval_results.json}
docker-compose.yml  Dockerfile.api  Dockerfile.ui
```

## Acknowledgments

CUAD v1 by The Atticus Project ([arXiv:2103.06268](https://arxiv.org/abs/2103.06268)).
