from langchain_nomic import NomicEmbeddings 
from .base import BaseEmbedings

class NomicEmbedding(BaseEmbedings):

    def __init__(self):
        self.model = NomicEmbeddings(
            model= "nomic-ai/nomic-embed-text-v1",
            # text_type="search_document"
        )
    def get_model(self):
        return self.model 
