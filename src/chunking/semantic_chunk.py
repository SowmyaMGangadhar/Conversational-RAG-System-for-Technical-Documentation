from langchain_experimental.text_splitter import (
    SemanticChunker
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_core.documents import Document


def semantic_chunk(docs):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    splitter = SemanticChunker(
        embeddings=embeddings
    )

    all_chunks = []

    for doc in docs:

        chunks = splitter.create_documents(
            [doc.page_content]
        )

        for chunk in chunks:

            all_chunks.append(
                Document(
                    page_content=chunk.page_content,
                    metadata=doc.metadata
                )
            )

    return all_chunks