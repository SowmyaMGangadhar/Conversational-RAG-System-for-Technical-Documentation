from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


bm25_cache = {}

def hybrid_search(vectorstore, chunks, query, k=5):

    global bm25_cache

    if "bm25" not in bm25_cache:
        print("Creating BM25 retriever...")
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_cache["bm25"] = bm25_retriever

    bm25_retriever = bm25_cache["bm25"]
    bm25_retriever.k = k

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_retriever
        ],
        weights=[0.5, 0.5])
    docs = ensemble_retriever.invoke(query)

    return docs


# from langchain_classic.retrievers import EnsembleRetriever
# from langchain_community.retrievers import BM25Retriever

# def hybrid_search(vectorstore, chunks, query, k=5):

#     bm25_retriever = BM25Retriever.from_documents(chunks)
#     bm25_retriever.k = k

#     vector_retriver = vectorstore.as_retriever(
#         search_kwargs={"k":k}
#     )

#     ensembel_retriver = EnsembleRetriever(
#         retrievers=[bm25_retriever, vector_retriver],
#         weights=[0.5, 0.5]
#             )
    
#     docs = ensembel_retriver.invoke(query)

#     return docs