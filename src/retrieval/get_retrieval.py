from .semantic_search import semantic_search
from .hybrid_search import hybrid_search
from .mmr_search import mmr_search
from .multi_query import multi_query_search
from .parent_child_retriever import parent_child_retriver
# from .contextual_compression import ContextualCompressionRetriever

def get_retriever(search_type:str):

    if search_type == "semantic":
        return semantic_search
    elif search_type == "hybrid":
        return hybrid_search
    elif search_type == "mmr":
        return mmr_search
    elif search_type == "multi_query":
        return multi_query_search
    elif search_type == "parent_child":
        return parent_child_retriver
    # elif search_type == "contextual":
    #     return ContextualCompressionRetriever
    else:
        raise ValueError(
            f"Unknown retriever type: {search_type}"
        )