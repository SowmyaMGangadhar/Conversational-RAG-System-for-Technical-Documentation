from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import uuid

def parent_child_chunk(docs):
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 200,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 400,
        chunk_overlap = 50,
    )

    parent_docs = []
    child_docs = []
    chunk_overlap = []

    for doc in docs:
        parent = parent_splitter.split_text(doc.page_content)
        for parent_text in parent:
            parent_id = str(uuid.uuid4())

            parent_doc = Document(
                page_content=parent_text,
                metadata={
                    "parent_id":parent_id,
                    **doc.metadata
                }
            )
            parent_docs.append(parent_doc)

            children = child_splitter.split_text(parent_text)

            for child_text in children:

                child_doc = Document(
                    page_content=child_text,
                    metadata={
                        "parent_id":parent_id,
                        **doc.metadata
                    }
                )

                child_docs.append(child_doc)
    return parent_docs, child_docs

