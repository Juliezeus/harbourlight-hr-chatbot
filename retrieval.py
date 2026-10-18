"""Local lexical baseline, or hosted dense + sparse retrieval with RRF."""
import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def lexical_search(query, docs, k=6):
    """Small dependency-free BM25 baseline. This is not semantic retrieval."""
    counts = {key: Counter(tokens(doc.page_content)) for key, doc in docs.items()}
    avg = sum(map(lambda c: sum(c.values()), counts.values())) / len(counts)
    scores = {}
    for key, count in counts.items():
        score = 0.0
        for term in set(tokens(query)):
            df = sum(term in c for c in counts.values())
            idf = math.log(1 + (len(counts) - df + 0.5) / (df + 0.5))
            tf = count[term]
            score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * sum(count.values()) / avg))
        scores[key] = score
    return [key for key in sorted(scores, key=scores.get, reverse=True) if scores[key] > 0][:k]


def rrf(*rankings):
    scores = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
    return sorted(scores, key=scores.get, reverse=True)


class PineconeHybrid:
    def __init__(self):
        from pinecone import Pinecone
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.namespace = os.getenv("PINECONE_NAMESPACE", "hds-hr-001-v1-1")
        self.indexes = [
            self.pc.Index(name=os.getenv("PINECONE_DENSE_INDEX", "harbourlight-dense")),
            self.pc.Index(name=os.getenv("PINECONE_SPARSE_INDEX", "harbourlight-sparse")),
        ]

    def search(self, query, docs):
        def search_one(index):
            result = index.search(
                namespace=self.namespace,
                query={"inputs": {"text": query}, "top_k": 6,
                       "filter": {"source_hash": {"$eq": next(iter(docs.values())).metadata["source_hash"]}}},
                fields=["section_id"],
            )
            return [hit["fields"]["section_id"] for hit in result["result"]["hits"]]
        with ThreadPoolExecutor(max_workers=2) as pool:
            rankings = list(pool.map(search_one, self.indexes))
        if not any(rankings):
            raise RuntimeError("No matching indexed corpus. Run ingestion and wait for indexing.")
        return rrf(*rankings)


def ingest(docs):
    """Explicit opt-in: creates hosted indexes and uploads synthetic policy only."""
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    namespace = os.getenv("PINECONE_NAMESPACE", "hds-hr-001-v1-1")
    configs = [
        (os.getenv("PINECONE_DENSE_INDEX", "harbourlight-dense"), "llama-text-embed-v2"),
        (os.getenv("PINECONE_SPARSE_INDEX", "harbourlight-sparse"), "pinecone-sparse-english-v0"),
    ]
    for name, model in configs:
        if not pc.has_index(name):
            pc.create_index_for_model(
                name=name, cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1"),
                embed={"model": model, "field_map": {"text": "chunk_text"}},
            )
        records = [
            {"_id": key, "chunk_text": doc.page_content, **doc.metadata}
            for key, doc in docs.items()
        ]
        pc.Index(name=name).upsert_records(namespace=namespace, records=records)
        print(f"Submitted {len(records)} sections to {name}/{namespace}")
