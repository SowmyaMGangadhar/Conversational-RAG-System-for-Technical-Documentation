from langchain_chroma import Chroma
from langchain_core.documents import Document
import os 

def chroms_vectorstore(
        chunks: list[Document],
        embedding_model,
        persist_directory: str = "data/vectorstores/chroma_db",
        collection_name: str = "rag_collection"):
    
    os.makedirs(persist_directory, exist_ok=True)

    vectorstores = Chroma.from_documents(
        documents=chunks,
        embedding = embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    print(f"Chroma DB saved at: {persist_directory}")
    return vectorstores
    