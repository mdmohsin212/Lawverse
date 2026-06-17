from __future__ import annotations
from typing import Iterable
from langchain_core.documents import Document

INSUFFICIENT_EVIDENCE_RESPONSE = (
    "### Answer\n"
    "The provided documents do not contain sufficient information to answer this question safely.\n\n"
    "### Sources\n"
    "No sufficiently relevant source was found in the indexed documents."
)

NON_LEGAL_RESPONSE = (
    "### Answer\n"
    "I'm designed to assist with Bangladeshi legal document questions. "
    "Please ask a legal question or upload/provide legal context."
)

GREETING_RESPONSE = (
    "### Answer\n"
    "Hello! I can help you ask questions about Bangladeshi legal documents and show sources from the retrieved context."
)

CLOSING_RESPONSE = (
    "### Answer\n"
    "You're welcome. Ask another legal-document question whenever you need help."
)


LEGAL_KEYWORDS = {
    "law", "legal", "court", "case", "act", "section", "rule", "rights", "contract",
    "agreement", "crime", "criminal", "civil", "penalty", "bail", "appeal", "property",
    "labour", "labor", "worker", "employee", "employer", "termination", "notice", "salary",
    "wage", "rent", "tax", "company", "constitution", "ordinance", "বাংলাদেশ", "আইন",
    "ধারা", "আদালত", "মামলা", "অধিকার", "শ্রম", "চুক্তি", "জামিন", "অপরাধ",
}

GREETING_WORDS = {"hi", "hello", "hey", "assalamu", "salam", "হাই", "হ্যালো", "সালাম"}
CLOSING_WORDS = {"thanks", "thank you", "bye", "goodbye", "ধন্যবাদ", "আচ্ছা", "বিদায়"}


def classify_simple_intent(text: str) -> tuple[str, str]:
    clean = (text or "").strip().lower()
    if not clean:
        return "empty", "Empty user input."

    token_hits = [kw for kw in LEGAL_KEYWORDS if kw in clean]
    if any(word in clean for word in GREETING_WORDS) and len(clean.split()) <= 8:
        return "greeting", "Short greeting detected."
    if any(word in clean for word in CLOSING_WORDS) and len(clean.split()) <= 10:
        return "closing", "Short closing/thanks message detected."
    if token_hits:
        return "legal_question", f"Legal keywords detected: {', '.join(token_hits[:5])}."

    if len(clean.split()) >= 8:
        return "legal_question", "Long-form question; routed to retrieval for evidence check."

    return "non_legal", "No legal intent signal detected."

def has_documents(docs: Iterable[Document]) -> bool:
    return bool(list(docs or []))