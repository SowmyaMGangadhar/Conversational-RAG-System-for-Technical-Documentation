from .bge_rerank import BGEReranker

def get_reranking(name: str):

    if name == "bge":
        return BGEReranker()
    else:
        raise ValueError(
            f"Unknown rerankers: {name}"
        )