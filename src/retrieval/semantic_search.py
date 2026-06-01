def semantic_search(vectorstore, query, k=5):
    retriver = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k":k})
    docs = retriver.invoke(query)

    return docs 