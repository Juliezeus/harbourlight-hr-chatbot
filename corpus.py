import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document

SOURCE = Path(__file__).resolve().parents[1] / "sources/harbourlight_people_and_reward_policy.md"
DEPENDENCIES = {
    "annual": ["LEA01", "BU02"],
    "personal": ["LEA02"],
    "parental": ["LEA03"],
    "compassionate": ["LEA04", "LEA02"],
    "wellbeing": ["BEN01", "BU03"],
    "learning": ["BEN02", "BU03"],
    "bonus": ["PAY02", "REW01", "REW02", "REW03", "BU04"],
    "salary": ["PAY01", "PAY02"],
    "promotion": ["REW04", "PAY02", "BU04"],
}


def load_corpus(path=SOURCE):
    text = Path(path).read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()
    parts = re.split(r"(?m)^## ([A-Z]+\d{2}) (.+)\n", text)
    docs = {}
    for pos in range(1, len(parts), 3):
        section, title, body = parts[pos:pos + 3]
        docs[section] = Document(
            page_content=f"{section} {title}\n{body.strip()}",
            metadata={
                "document_id": "HDS-HR-001", "version": "1.1",
                "section_id": section, "title": title, "country": "AU",
                "effective_from": "2026-10-01", "source_hash": digest,
                "audience": "all_employees", "source": str(path),
                # BU sections contain multiple units, so preserve all candidates.
                "applicable_bus": ["ALL", "CORP", "TECH", "OPS"],
            },
        )
    if len(docs) != 22:
        raise ValueError("Expected the approved 22-section v1.1 synthetic corpus")
    return docs


def evidence_ids(topics, ranked, transfer=False):
    """Required evidence cannot be displaced by similarity ranking."""
    ids = ["GOV02", "BU01"]
    for topic in topics:
        ids.extend(DEPENDENCIES.get(topic, []))
    if transfer:
        ids.append("BU05")
    ids.extend(ranked[:3])
    return list(dict.fromkeys(ids))
