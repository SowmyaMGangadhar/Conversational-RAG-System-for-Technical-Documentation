from .bge_embeddings import BGEEmbeddings
from .e5_embeddings import E5Embeddings
from .nomic_embeddings import NomicEmbedding
from .instructor_embeddings import InstructorEmbedding

def get_embeddings(model_name: str):

    if model_name == "bge":
        return BGEEmbeddings().get_model()
    elif model_name == "e5":
        return E5Embeddings().get_model()
    elif model_name == "nomic":
        return NomicEmbedding().get_model()
    elif model_name == "instructor":
        return InstructorEmbedding().get_model()
    else:
        raise ValueError(f"Unknown embedding model: {model_name}")