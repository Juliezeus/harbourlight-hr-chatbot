# Harbourlight HR policy RAG prototype

A Python command-line prototype for the fictional Harbourlight People and Reward Policy Corpus v1.1: one Markdown source with 22 policy sections. The intended future Confluence corpus is not connected.

## Deploy a shareable Streamlit app

1. Upload the project source to a GitHub repository. Include `chat_app.py`, `requirements.txt`, `hr_rag/`, `sources/`, and the optional evaluation, tests and documentation. Never upload `.env`, `.venv`, or `.streamlit/secrets.toml`.
2. Sign in at https://share.streamlit.io and choose Create app, then select the repository, branch and main file `chat_app.py`.
3. In Advanced settings choose Python 3.13 and paste the TOML from `.streamlit/secrets.example.toml`, replacing only the two key placeholders with real keys from your local `.env`. The example file itself contains no secrets. Keys must remain in deployment settings, not GitHub files or chat.
4. Deploy. Community Cloud supplies the shareable URL after a successful build. Test a question and its source expanders in the deployed app.

The deployment reuses existing Pinecone indexes. No ingestion command runs during app startup. Use the same corpus file as the indexed copy because retrieval filters by source hash. This synthetic demonstration has no employee authentication. Shared use consumes the configured Pinecone and Nebius accounts. A cloud URL is not available until the repository is connected and deployment succeeds.

## Architecture

LangChain Documents and prompts feed a LangGraph workflow:

`extract context -> clarify or retrieve -> expand policy dependencies -> generate -> verify -> return`

Live retrieval runs Pinecone hosted dense and sparse searches concurrently, merges rankings with reciprocal rank fusion, and fetches required company defaults and BU overrides from the canonical local corpus. Both indexes use integrated embedding. This uses the vector-records API rather than the Documents API. Local mode is a BM25 lexical baseline only; it does not simulate embeddings or generate answers.

Company-wide and BU sections remain available together. All source sections are employee-readable, so this prototype does not implement production authentication or permission enforcement. Employee profile flags are user-supplied context, never authorisation.

## Local quick start

### Browser chat

```powershell
.\.venv\Scripts\python -m streamlit run chat_app.py --server.address 127.0.0.1
```

Open http://localhost:8501. The sidebar provides optional BU and FTE context. Generate answers uses the configured Pinecone and LLM services; switch it off for an API-free evidence preview. Sources can be expanded beneath answers. Conversation history is limited to the six most recent messages supplied to the model and is held only in that browser session. New conversation clears that history; sidebar context remains selected. Credentials stay in server-side `.env`. This is a local synthetic-policy demo without employee authentication. Do not deploy it as a real employee service before adding access controls.

Run from the project directory with Python 3.11 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m hr_rag inspect
.\.venv\Scripts\python -m hr_rag ask "I am a 0.6 FTE Technology employee. What are my wellbeing and learning budgets?"
.\.venv\Scripts\python -m unittest discover -s tests -v
```

The ask command returns JSON with extracted context, section IDs, evidence text and stage timings. Local mode deliberately returns `evidence_only`, not a policy answer. Its small keyword parser is for smoke tests; natural-language extraction is enabled in live mode.

## Live setup

### Local Ollama answer model

The current `.env.example` defaults to local Ollama using ChatOpenAI's OpenAI-compatible transport. Install [Ollama for Windows](https://ollama.com/download/windows), start the Ollama application, and download the configured model:

```powershell
ollama pull qwen2.5:7b
ollama list
```

If the background service is not running, run `ollama serve` in a separate terminal. Configuration:

```dotenv
LLM_BACKEND=ollama
CHAT_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434/v1/
LLM_TIMEOUT_SECONDS=120
```

The client supplies `api_key="ollama"`, which local Ollama ignores. No OpenAI credits or key are used for this backend. JSON-schema structured output is requested explicitly for extraction, answers and review. The model choice is a starting point, not a validated quality or hardware recommendation. Change CHAT_MODEL to an installed compatible model if necessary. Local model loading and inference may exceed the eight-second target; the 120-second request timeout is operational tolerance, not a revised latency target.

Pinecone still provides remote dense/sparse retrieval and embeddings, so `PINECONE_API_KEY` and ingestion remain necessary for `ask --live`. Questions and answer generation stay on the local Ollama endpoint; search query text still goes to Pinecone. Local mode without `--live` remains evidence-only.

### Hosted OpenAI alternative

To use hosted OpenAI instead, set `LLM_BACKEND=openai`, populate `OPENAI_API_KEY`, and set `CHAT_MODEL` to a provider:model identifier such as `openai:<available-model-id>`. Keep secrets out of chat and version control. The supplied dependency list includes the OpenAI integration; other providers need their integration package and credentials.

Choose the Pinecone cloud and region before ingestion. Defaults are examples, not a data-residency decision. Then run:

```powershell
.\.venv\Scripts\python -m hr_rag ingest --live
.\.venv\Scripts\python -m hr_rag ask "What learning budget applies to a Technology employee?" --bu TECH --live
```

Ingestion explicitly creates two indexes if absent and sends the synthetic policy to Pinecone. Generation sends the question, supplied context and retrieved synthetic policy to the configured model provider. These operations may incur charges. Index readiness or eventual indexing can require retrying later; no silent local fallback occurs. Existing indexes must use the expected models and `chunk_text` field mapping. Changing source text requires re-ingestion: searches filter by its SHA-256 hash to avoid mixing corpus revisions. Stale remote records are not automatically deleted.

## Context and grounding

- Unknown employee facts stay unknown; live extraction asks for material missing context.
- A project assignment does not establish home BU; part-time days do not automatically establish FTE.
- Required evidence bundles preserve rules such as BEN02 plus BU03 for learning and LEA01 plus BU02 for annual leave.
- Transfers add BU05; bonus questions retain company eligibility, calculations and BU targets.
- Generated claims require valid retrieved section citations and a separate model review for support, applicability and completeness. Failed reviews return a referral instead of the draft.
- Personal calculations are deferred to Payroll in this first release. Deterministic calculation tools and richer historical employee context are next steps.
- No conversational persistence is enabled yet. Re-submit the question with clarifying context or use `--bu` and `--fte`.

## Evaluation and limits

Targets are at least 95% supported substantive claims and p95 completed-response latency at most 8 seconds. Neither target has been demonstrated. Model self-review is a runtime check, not an independent faithfulness measurement.

Local tests check evidence coverage, dependency retention, basic context parsing and fusion. They do not measure answer quality, production retrieval quality or live API compatibility. Live generation has three sequential model calls, so latency may exceed the target. Stage timings and a per-request latency flag expose this; there is not yet an overall deadline or reranking stage.

Before reporting success, create a held-out, human-reviewed evaluation set covering BU defaults, part-time conditions, transfers, missing inputs, excluded workers, historical dates and unsupported topics. Annotate required sections, supported claims, expected clarifications and answer completeness. Compare dense-only, hybrid, and hybrid plus dependency expansion. Report retrieval and end-to-end latency separately, with sample count and failure cases.

Further production work includes topic-specific BU chunks, effective-date intervals, authenticated employee profiles, access enforcement, persistent clarification state, deterministic calculations, timeout handling and independently scored evaluation.

## Framework references

- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [Pinecone hybrid search](https://docs.pinecone.io/guides/search/hybrid-search)
- [Pinecone integrated embedding index creation](https://docs.pinecone.io/guides/index-data/create-an-index)
