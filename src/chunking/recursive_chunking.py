from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_core.documents import Document


def recursive_chunk(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []

    for doc in docs:

        chunks = splitter.split_text(
            doc.page_content
        )

        for chunk in chunks:

            all_chunks.append(
                Document(
                    page_content=chunk,
                    metadata=doc.metadata
                )
            )

    return all_chunks