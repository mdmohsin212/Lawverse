LEGAL_DISCLAIMER = (
    "Lawverse provides educational legal information from retrieved documents. "
    "It is not a substitute for a licensed lawyer."
)


def append_legal_disclaimer(answer: str, include: bool = False) -> str:
    answer = (answer or "").strip()
    
    if not include:
        return answer
    if not answer:
        return f"> **Note:** {LEGAL_DISCLAIMER}"

    if "not a substitute for a licensed lawyer" in answer.lower():
        return answer

    return f"{answer}\n\n> **Note:** {LEGAL_DISCLAIMER}"