LEGAL_DISCLAIMER = (
    "Lawverse is an educational legal information assistant. "
    "It is not a substitute for a licensed lawyer."
)

def append_legal_disclaimer(answer: str) -> str:
    answer = (answer or "").strip()
    if not answer:
        return f"### Legal Disclaimer\n{LEGAL_DISCLAIMER}"

    if "not a substitute for a licensed lawyer" in answer.lower():
        return answer

    return f"{answer}\n\n### Legal Disclaimer\n{LEGAL_DISCLAIMER}"