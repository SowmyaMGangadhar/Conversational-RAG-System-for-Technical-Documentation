from .ollama_generation import OllamaGeneration

def get_generation_model(model_name):

    if model_name == "ollama":
        return OllamaGeneration().get_llm()
    else:
        raise ValueError(
            f"Unkown generation model: {model_name}"
            
        )
