def mmr_search(vectorstore, query, k=5, fetch_k=20):

    retriever = vectorstore.as_retriever(
        search_type = "mmr",
        search_kwargs = {
            "k":k,
            "fetch_k":fetch_k})

    docs = retriever.invoke(query)

    return docs 