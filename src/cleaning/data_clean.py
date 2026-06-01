import re
from bs4 import BeautifulSoup

SKIP_EXACT = {
    "Was this page helpful",
    "Was this page helpful?",
    "Skip to main content",
    "Edit this page on GitHub",
    "Connect these docs",
    "Yes",
    "No",
    "Next",
    "Previous",
    "Documentation Index",
}

SKIP_CONTAINS = [
    "fetch the complete documentation index at:",
    "use this file to discover all available pages before exploring further.",
]

SKIP_PATTERNS = [
    re.compile(r"^[⌘⌥⇧⌃↑↓←→✕×]+$"),
    re.compile(r"^⌘\s*[A-Z]$"),
]

def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    tab_panels = soup.find_all(attrs={"role": "tabpanel"})
    for panel in tab_panels[1:]:
        panel.decompose()

    for tag in soup.find_all(attrs={"role": ["tab", "tablist"]}):
        tag.decompose()

    for pre in soup.find_all("pre"):
        code_text = pre.get_text(" ")
        code_text = re.sub(r"[ \t]+", " ", code_text).strip()
        pre.replace_with("\n" + code_text + "\n")

    for tag in soup.find_all(["a", "code", "em", "strong", "b", "i", "span"]):
        tag.replace_with(tag.get_text(" "))

    full_text = soup.get_text("\n")
    raw_lines = [line.strip() for line in full_text.split("\n") if line.strip()]

    cleaned = []
    for line in raw_lines:
        line = re.sub(r"\s+", " ", line).strip()

        if len(line) < 2:
            continue
        if line in SKIP_EXACT:
            continue
        if any(s in line.lower() for s in SKIP_CONTAINS):
            continue
        if any(p.match(line) for p in SKIP_PATTERNS):
            continue

        cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_true_index_page(doc, cleaned_text: str) -> bool:
    url = doc.metadata.get("source", "").lower()

    index_like_urls = [
        "llms.txt",
        "genindex",
        "/index.html",
    ]

    if any(x in url for x in index_like_urls):
        return True

    lines = [l.strip() for l in cleaned_text.splitlines() if l.strip()]
    if len(lines) <= 5 and "index" in cleaned_text.lower():
        return True

    return False


def clean_docs(docs, min_len=120):
    cleaned_docs = []

    for doc in docs:
        source = doc.metadata.get("source_docs", "unknown")
        url = doc.metadata.get("source", "")

        original_text = doc.page_content or ""
        cleaned_text = clean_text(original_text)

        if is_true_index_page(doc, cleaned_text):
            print(f"[DROP-INDEX] [{source}] url={url}")
            continue

        if len(cleaned_text) < min_len:
            print(f"[DROP-SHORT] [{source}] len={len(cleaned_text)} url={url}")
            print(f"[DROP-SHORT] preview: {repr(cleaned_text[:300])}")
            continue

        doc.page_content = cleaned_text
        cleaned_docs.append(doc)

    return cleaned_docs


# def clean_docs(docs):
#     cleaned_docs = []

#     noisy_patterns = [
#         "llms.txt",
#         "documentation index",
        
#     ]

#     for doc in docs:
#         cleaned_text = clean_text(doc.page_content)
#         lower_text = cleaned_text.lower()

#         if any(pattern in lower_text for pattern in noisy_patterns):
#             continue

#         if len(cleaned_text) < 200:
#             continue

#         doc.page_content = cleaned_text
#         cleaned_docs.append(doc)

#     return cleaned_docs