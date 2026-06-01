from langchain_text_splitters import CharacterTextSplitter

def fixed_chunking(docs):
    splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap = 200,)
    chunks = splitter.split_documents(docs)
    return chunks
    