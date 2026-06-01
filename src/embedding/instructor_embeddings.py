from langchain_huggingface import HuggingFaceEmbeddings
from .base import BaseEmbedings

class InstructorEmbedding(BaseEmbedings):

    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name="hkunlp/instructor-xl"
        )
    
    def get_model(self):
        return self.model