from langchain_ollama import ChatOllama

class OllamaGeneration:
    def __init__(self, model_name="qwen2.5:7b"):
        self.llm = ChatOllama(
            model = model_name, temperature=0)
    def get_llm(self):
        return self.llm