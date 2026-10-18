import argparse
import json
import os
import time
from pathlib import Path

from .corpus import load_corpus
from .retrieval import PineconeHybrid, ingest
from .workflow import build_workflow, chat_model


def main():
    # Use OS-managed trust roots, including Windows enterprise certificates.
    # Certificate and hostname verification remain enabled.
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", encoding="utf-8-sig")
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description="Harbourlight HR RAG prototype")
    parser.add_argument("command", choices=["inspect", "ingest", "ask"])
    parser.add_argument("question", nargs="?", default="")
    parser.add_argument("--live", action="store_true", help="Use Pinecone retrieval and the configured hosted or local LLM")
    parser.add_argument("--bu", choices=["CORP", "TECH", "OPS"])
    parser.add_argument("--fte", type=float)
    args = parser.parse_args()
    if args.live and not os.getenv("PINECONE_API_KEY", "").strip():
        parser.error("PINECONE_API_KEY is empty or missing. Add it to the project's .env file (not .env.example), save, and retry. Never paste your key into chat.")
    if args.command == "ask" and args.live:
        if os.getenv("LLM_BACKEND", "openai").lower() == "nebius" and not os.getenv("NEBIUS_API_KEY", "").strip():
            parser.error("NEBIUS_API_KEY is empty or missing. Add your Nebius Token Factory key to .env and save.")
        model_name = os.getenv("CHAT_MODEL", "").strip()
        if not model_name:
            parser.error("CHAT_MODEL is empty. Set a provider:model identifier in the project's .env file.")
        if os.getenv("LLM_BACKEND", "openai").lower() not in {"ollama", "nebius"} and (model_name.startswith("openai:") or model_name.startswith("gpt-")) and not os.getenv("OPENAI_API_KEY", "").strip():
            parser.error("OPENAI_API_KEY is empty or missing. Pinecone provides retrieval embeddings; the configured OpenAI answer model needs a separate key. Add it to .env, save, and retry.")
    if args.fte is not None and not 0 < args.fte <= 1:
        parser.error("FTE must be greater than zero and no more than one")
    docs = load_corpus()
    if args.command == "inspect":
        print(json.dumps({k: {"words": len(d.page_content.split()), **d.metadata} for k, d in docs.items()}, indent=2))
        return
    if args.command == "ingest":
        if not args.live:
            parser.error("Use --live to create indexes and upload the synthetic corpus")
        ingest(docs)
        return
    if not args.question.strip():
        parser.error("Supply a question")
    start = time.perf_counter()
    profile = {key: value for key, value in {"bu": args.bu, "fte": args.fte}.items() if value is not None}
    app = build_workflow(docs, PineconeHybrid() if args.live else None, chat_model() if args.live else None)
    try:
        result = app.invoke({"question": args.question, "profile": profile})
    except Exception as exc:
        if os.getenv("LLM_BACKEND", "openai").lower() == "nebius" and getattr(exc, "status_code", None) == 404:
            parser.exit(1, "The configured Nebius model is not available. Check the model ID against your Token Factory model catalog.\n")
        if os.getenv("LLM_BACKEND", "openai").lower() == "ollama" and type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}:
            parser.exit(1, "Cannot reach Ollama or the request timed out. Start Ollama, check OLLAMA_BASE_URL, "
                        "and run 'ollama list' to confirm CHAT_MODEL is installed. "
                        "Cold model loading may require a longer LLM_TIMEOUT_SECONDS.\n")
        body = getattr(exc, "body", {})
        error = body.get("error", body) if isinstance(body, dict) else {}
        if error.get("code") == "credit_balance_exhausted":
            parser.exit(1, "OpenAI API credits are exhausted for the organisation associated with your key. "
                        "Check https://platform.openai.com/settings/organization/billing/ and add API credits. "
                        "Retry after the balance updates. Your Pinecone data does not need re-ingestion.\n")
        raise
    result["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    result["latency_target_met"] = result["elapsed_seconds"] <= 8
    if not args.live:
        result["evidence_preview"] = {key: docs[key].page_content for key in result["evidence"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
