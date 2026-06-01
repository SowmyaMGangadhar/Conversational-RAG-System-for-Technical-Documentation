from langchain_classic.retrievers.multi_query import MultiQueryRetriever

def multi_query_search(vectorstore, llm, query):

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),llm=llm)

    docs = retriever.invoke(query)

    return docs