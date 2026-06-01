import re
from difflib import SequenceMatcher

STOPWORDS = {
    "the", "is", "a", "an", "and", "or", "but", "in", "on",
    "at", "to", "for", "of", "with", "by", "from", "it",
    "this", "that", "are", "was", "be", "as", "have", "has"
}

def _content_words(text: str) -> set:
    """Lowercase, remove punctuation, filter stopwords."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return {t for t in tokens if t not in STOPWORDS}


def faithfulness_score(response: str, context: str) -> float:
    """
    Sentence-level: what fraction of response sentences
    have at least one supporting sentence in context?
    
    Uses SequenceMatcher correctly — sentence vs sentence,
    not full string vs full string.
    """
    def split_sentences(text):
        return [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 10]

    response_sentences = split_sentences(response)
    context_sentences  = split_sentences(context)

    if not response_sentences:
        return 0.0

    supported = 0
    for r_sent in response_sentences:
        # Check if any context sentence is sufficiently similar
        best_match = max(
            SequenceMatcher(None, r_sent.lower(), c_sent.lower()).ratio()
            for c_sent in context_sentences
        ) if context_sentences else 0.0

        if best_match >= 0.5:   # tune this threshold
            supported += 1

    return round(supported / len(response_sentences), 4)


def groundedness_score(response: str, context: str) -> float:
    """
    What fraction of meaningful response words appear in context?
    Filters stopwords so common filler words don't skew results.
    """
    response_words = _content_words(response)
    context_words  = _content_words(context)

    if not response_words:
        return 0.0

    overlap = response_words.intersection(context_words)
    return round(len(overlap) / len(response_words), 4)


def hallucination_score(response: str, context: str) -> float:
    """
    What fraction of meaningful response words are NOT in context?
    Genuinely inverse of groundedness (both kept for explicitness).
    """
    # This is intentionally 1 - groundedness, but now meaningful
    # because stopwords are removed — only content words matter
    return round(1.0 - groundedness_score(response, context), 4)