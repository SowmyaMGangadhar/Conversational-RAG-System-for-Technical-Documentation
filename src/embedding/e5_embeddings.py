from langchain_huggingface import HuggingFaceEmbeddings
from .base import BaseEmbedings

class E5Embeddings(BaseEmbedings):

    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name= "intfloat/e5-base-v2",
            encode_kwargs={"normalize_embeddings": True}

        )
    def get_model(self):
        return self.model