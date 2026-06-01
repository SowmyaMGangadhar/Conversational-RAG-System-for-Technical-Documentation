import time 
from src.rag_chain import RAGPipeline
from src.evaluation.retrieval_eval import precision_at_k, recall_at_k, hit_rate
from src.evaluation.geenrator_eval import faithfulness_score, groundedness_score, hallucination_score

test_cases = [

    {
        "query": "What is async and await in FastAPI?",
        "relevant_ids": [
            "fastapi_0daeb4b693d7",
            "fastapi_1dc9528005a1",
            "fastapi_f4abb06e0094",
            "fastapi_e4b3ba523727",
            "fastapi_e9cfd49a25e9"
        ]
    },

    {
        "query": "How to execute ONNX models using ONNX Runtime?",
        "relevant_ids": [
            "pytorch_f349861aa2be",
            "pytorch_1f437dece16f",
            "pytorch_c64002f7ebf6",
            "pytorch_9527a7160733",
            "pytorch_2087588a656e"
        ]
    },

    {
        "query": "What is quantization aware training?",
        "relevant_ids": [
            "tensorrt_437b7d76df7e",
            "tensorrt_0594e674c7ec",
            "tensorrt_d651d278bbce",
            "tensorrt_5b7457c6a8b7"
        ]
    },

    {
        "query": "How to create a deep agent?",
        "relevant_ids": [
            "langchain_626410475d7f",
            "langchain_0b0c9481f163",
            "langchain_ef4e1285d828",
            "langchain_4a1a332cbd36",
            "langchain_2c8b736bae69"
        ]
    }

]

pipeline = RAGPipeline(
    chunk_type="recursive",
    embedding_type="bge",
    retrieval_type="hybrid",
    reranker_type="bge",
    generation_model="ollama"
)

for test in test_cases:
    query = test["query"]
    relevant_ids = test["relevant_ids"]

    start_time = time.time()

    result = pipeline.run(query)

    latency = time.time() - start_time

    response = result["response"]
    context = result["context"]
    docs = result["retrieved_docs"]

    retrieved_ids = [doc.metadata["doc_id"] for doc in docs]

    # retrieval metrics

    precision = precision_at_k(retrieved_ids, relevant_ids, k=5)
    recall = recall_at_k(retrieved_ids, relevant_ids, k=5)
    hitrate = hit_rate(retrieved_ids, relevant_ids, k=5)

    # generation metrics

    faithfulness = faithfulness_score(response, context)
    groundedness = groundedness_score(response, context)
    hallucination = hallucination_score(response, context)



    print(f"\nQUERY:\n{query}")
    print("\nRESPONSE:\n")
    print(response)
    print("\nRETRIEVED IDS:")
    print(retrieved_ids)
    print("RELEVANT IDS:")
    print(relevant_ids)
    print("\nRETRIEVAL METRICS\n")

    print(f"Precision@5 : {precision:.4f}")
    print(f"Recall@5    : {recall:.4f}")
    print(f"HitRate@5   : {hitrate}")

    print("\nGENERATION METRICS\n")

    print(f"Faithfulness   : {faithfulness}")
    print(f"Groundedness   : {groundedness}")
    print(f"Hallucination  : {hallucination}")

    print(f"\nLatency : {latency:.4f} sec")



