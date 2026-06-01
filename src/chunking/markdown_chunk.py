from langchain_text_splitters import (
    MarkdownHeaderTextSplitter
)

from langchain_core.documents import Document


def markdown_chunk(docs):

    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    all_chunks = []

    for doc in docs:

        chunks = splitter.split_text(
            doc.page_content
        )

        for chunk in chunks:

            all_chunks.append(
                Document(
                    page_content=chunk.page_content,
                    metadata={
                        **doc.metadata,
                        **chunk.metadata
                    }
                )
            )

    return all_chunks