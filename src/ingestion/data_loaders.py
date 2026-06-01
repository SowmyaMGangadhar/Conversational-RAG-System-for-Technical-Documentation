from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

DOC_SOURCES = {
    "langchain" : "https://docs.langchain.com/oss/python/deepagents/overview",
    "pytorch" : "https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html",
    "fastapi" : "https://fastapi.tiangolo.com/async/",
    "tensorrt" : "https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html"

}

MAX_DOCS_PER_SOURCE = 10
def load_docs(doc_names: str):
    all_docs = []
    for name in doc_names:
        print(f"\n Loading {name} docs...")
        loader = RecursiveUrlLoader(
        url=DOC_SOURCES[name],
        max_depth=2,        
        prevent_outside=True,)

        docs = loader.load()
        docs = docs[:MAX_DOCS_PER_SOURCE]

        for doc in docs:
            doc.metadata["source_docs"] = name
            print(f"Loaded {len(docs)} docs")
        all_docs.extend(docs)
    print(f"Toal docs loaded: {len(all_docs)}")
        


    return all_docs
