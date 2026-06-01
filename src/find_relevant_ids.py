from rag_chain import RAGPipeline

pipeline = RAGPipeline(
    chunk_type="recursive",
    embedding_type="bge",
    retrieval_type="hybrid",
    reranker_type="bge",
    generation_model="ollama"
)

queries = [

    "What is async and wait in fastapi?",
    "How to execute the ONNX model with ONNX runtime?",
    "what is quantization aware training?",
    "How to create a deep agent?"
]

for query in queries:
    result = pipeline.run(query)
    print(f"\nQUERY: {query}")
    for doc in result["retrieved_docs"]:
        print(f"  ID : {doc.metadata['doc_id']}")
        print(f"  ID : {doc.metadata['source_docs']}")
        print(f"  TEXT: {doc.page_content[:150]}")
        print()

# from ingestion.data_loaders import load_docs
# from cleaning.data_clean import clean_docs
# from collections import Counter

# docs = load_docs(["langchain", "fastapi", "pytorch", "tensorrt"])
# cleaned = clean_docs(docs)

# counts = Counter(doc.metadata["source_docs"] for doc in cleaned)
# print("\nDocs per source after cleaning:")
# for source, count in counts.items():
#     print(f"  {source:12} : {count} docs")

# You want all 4 showing non-zero counts before running find_relevant_ids