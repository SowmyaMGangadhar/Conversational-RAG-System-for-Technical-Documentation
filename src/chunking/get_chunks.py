from .fixed_size import fixed_chunking
from .recursive_chunking import recursive_chunk
from .markdown_chunk import markdown_chunk
from .semantic_chunk import semantic_chunk
from .parent_chld_chunk import parent_child_chunk

def get_chunks(chunk_type: str, cleaned_docs):
    if chunk_type == "fixed":
        return fixed_chunking(cleaned_docs)
    elif chunk_type == "recursive":
        return recursive_chunk(cleaned_docs)
    elif chunk_type == "semantic":
        return semantic_chunk(cleaned_docs)
    elif chunk_type == "markdown":
        return markdown_chunk(cleaned_docs)
    elif chunk_type == "parent-child":
        return parent_child_chunk(cleaned_docs)
    else:
        raise ValueError (
            f"Unknown chunk type: {chunk_type}"
        )