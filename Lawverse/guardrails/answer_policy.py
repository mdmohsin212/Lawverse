from __future__ import annotations
import re
from typing import Iterable
from langchain_core.documents import Document

INSUFFICIENT_EVIDENCE_RESPONSE = (
    "I could not find enough information in the uploaded legal documents to answer this safely.\n\n"
    "**Sources**\n\n"
    "No sufficiently relevant source was found in the indexed documents."
)

NON_LEGAL_RESPONSE = (
    "I'm designed to assist with Bangladeshi legal document questions. "
    "Please ask a legal question or provide legal context."
)

GREETING_RESPONSE = (
    "Hello! I can help you ask questions about Bangladeshi legal documents "
    "and show sources when I use retrieved context."
)

CLOSING_RESPONSE = "You're welcome. Ask another legal-document question whenever you need help."


LEGAL_KEYWORDS = {
    "law", "legal", "court", "case", "act", "section", "rule", "rights", "contract",
    "agreement", "crime", "criminal", "civil", "penalty", "bail", "appeal", "property",
    "labour", "labor", "worker", "employee", "employer", "termination", "notice", "salary",
    "wage", "rent", "tax", "company", "constitution", "ordinance", "maternity", "retrenchment",
    "dismissal", "compensation", "digital", "security", "cyber", "fraud", "hacking",
    "বাংলাদেশ", "আইন", "ধারা", "আদালত", "মামলা", "অধিকার", "শ্রম", "চুক্তি", "জামিন",
    "অপরাধ", "কোম্পানি", "শ্রমিক", "নিয়োগ", "বরখাস্ত", "ক্ষতিপূরণ",
}

LEGAL_PHRASES = {
    "labour law", "labor law", "digital security", "companies act", "company law",
    "bangladesh labour", "bangladesh labor", "digital security act", "labour act",
}

GREETING_WORDS = {"hi", "hello", "hey", "assalamu", "salam", "হাই", "হ্যালো", "সালাম"}
CLOSING_WORDS = {"thanks", "thank", "bye", "goodbye", "ধন্যবাদ", "আচ্ছা", "বিদায়"}

def _tokens(text: str) -> list[str]:
    english = re.findall(r"[a-zA-Z]+", text.lower())
    bangla = re.findall(r"[\u0980-\u09FF]+", text)
    return english + bangla


def _has_legal_signal(clean: str, toks: list[str]) -> tuple[bool, str]:
    token_set = set(toks)
    token_hits = sorted(LEGAL_KEYWORDS.intersection(token_set))
    phrase_hits = sorted([phrase for phrase in LEGAL_PHRASES if phrase in clean])

    if phrase_hits:
        return True, f"Legal phrases detected: {', '.join(phrase_hits[:5])}."
    if token_hits:
        return True, f"Legal keywords detected: {', '.join(token_hits[:5])}."
    return False, "No legal keyword or phrase found."


def _is_short_greeting(clean: str, toks: list[str]) -> bool:
    token_set = set(toks)
    if not token_set.intersection(GREETING_WORDS):
        return False

    non_greeting_tokens = [
        t for t in toks
        if t not in GREETING_WORDS and t not in {"bro", "there", "lawverse"}
    ]
    return len(toks) <= 4 and len(non_greeting_tokens) <= 2


def _is_short_closing(clean: str, toks: list[str]) -> bool:
    token_set = set(toks)
    if not token_set.intersection(CLOSING_WORDS):
        return False
    return len(toks) <= 5


def classify_simple_intent(text: str) -> tuple[str, str]:
    clean = (text or "").strip().lower()
    if not clean:
        return "empty", "Empty user input."

    toks = _tokens(clean)

    has_legal, legal_reason = _has_legal_signal(clean, toks)
    if has_legal:
        return "legal_question", legal_reason

    if _is_short_greeting(clean, toks):
        return "greeting", "Short greeting detected."

    if _is_short_closing(clean, toks):
        return "closing", "Short closing/thanks message detected."

    if len(toks) >= 8:
        return "legal_question", "Long-form question; routed to retrieval for evidence check."

    return "non_legal", "No legal intent signal detected."

def has_documents(docs: Iterable[Document]) -> bool:
    return bool(list(docs or []))