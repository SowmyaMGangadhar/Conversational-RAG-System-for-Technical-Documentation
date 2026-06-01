from langchain_huggingface import HuggingFaceEmbeddings
from .base import BaseEmbedings

class BGEEmbeddings(BaseEmbedings):

    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name = "BAAI/bge-small-en-v1.5",
            encode_kwargs={"normalize_embeddings":True}
        )
    
    def get_model(self):
        return self.model