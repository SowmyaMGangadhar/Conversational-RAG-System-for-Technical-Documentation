from openai import OpenAI

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
)

from langchain_ollama import OllamaEmbeddings

from src.rag_test_cases import TEST_CASES
from src.rag_chain import RAGPipeline
import logging
import pandas as pd

logging.getLogger("ragas").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def run_ragas_evaluation():
    pipeline = RAGPipeline()

    ollama_client = OpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    )

    ragas_llm = llm_factory(
        "llama3.1",
        provider="openai",
        client=ollama_client,
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="nomic-embed-text")
    )

    metrics = [
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        Faithfulness(llm=ragas_llm),
    ]

    samples = []

    for test in TEST_CASES[:2]:
        print(f"\nRunning: {test['query']}")
        result = pipeline.run(test["query"])

        response = (result.get("response") or "").strip()
        contexts = [
            (doc.page_content or "").strip()
            for doc in result.get("reranked_docs", [])
            if (doc.page_content or "").strip()
        ]

        print(f"  response_len={len(response)}")
        print(f"  context_count={len(contexts)}")

        if not response:
            print("  [SKIP] Empty response")
            continue

        if not contexts:
            print("  [SKIP] No retrieved contexts")
            continue

        samples.append(
            SingleTurnSample(
                user_input=test["query"],
                response=response,
                retrieved_contexts=contexts,
                reference=test["ground_truth"].strip(),
            )
        )

    if not samples:
        raise ValueError("No valid samples were created for evaluation.")

    dataset = EvaluationDataset(samples=samples)

    run_config = RunConfig(
        timeout=900,
        max_workers=1,
        max_retries=3,
        max_wait=60,
    )

    scores = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
    )

    df = scores.to_pandas()

    metric_cols = ["answer_relevancy", "faithfulness"]
    existing_cols = [c for c in metric_cols if c in df.columns]

    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)

    print("\nPer-Query Scores:")
    print("-" * 60)
    for i, row in df.iterrows():
        print(f"\nQuery {i+1}: {samples[i].user_input[:60]}...")
        for col in existing_cols:
            val = row[col]
            if pd.notna(val):
                print(f"  {col:<20}: {val:.4f}")
            else:
                print(f"  {col:<20}: NaN")

    print("\n" + "-" * 60)
    print("Average Scores:")
    print("-" * 60)
    for col in existing_cols:
        avg = df[col].mean()
        if pd.notna(avg):
            print(f"  {col:<20}: {avg:.4f}")
        else:
            print(f"  {col:<20}: NaN")

    print("=" * 60)

    return scores


if __name__ == "__main__":
    run_ragas_evaluation()