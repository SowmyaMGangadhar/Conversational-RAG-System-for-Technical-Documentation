from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os


def faiss_vectorstore(
    chunks: list[Document],
    embedding_model,
    save_path: str = "data/vectorstores/faiss_index"
):

    if os.path.exists(save_path):

        vectorstore = FAISS.load_local(
            save_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        print(f"Loaded existing FAISS index from: {save_path}")

    else:

        os.makedirs(save_path, exist_ok=True)

        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

        vectorstore.save_local(save_path)

        print(f"FAISS index saved at: {save_path}")

    return vectorstore

# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# import os 


# def faiss_vectorstore(chunks:list[Document],
#                       embedding_model,
#                       save_path: str = "data/vectorstores/faiss_index"):
#     os.makedirs(save_path, exist_ok=True)

#     vectorstore = FAISS.from_documents(
#         documents=chunks,
#         embedding=embedding_model
#     )

#     vectorstore.save_local(save_path)
#     print(f"FAISS index saved at: {save_path}")

#     return vectorstore