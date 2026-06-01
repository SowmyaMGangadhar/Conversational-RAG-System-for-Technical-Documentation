import os
import hashlib
import pickle
from src.ingestion.data_loaders import load_docs
from src.cleaning.data_clean import clean_docs
from src.chunking.get_chunks import get_chunks
from src.embedding.get_embeddings import get_embeddings
from src.vectorstore.faiss import faiss_vectorstore
from src.retrieval.get_retrieval import get_retriever
from src.reranking.get_reranking import get_reranking
from src.generation.get_generation import get_generation_model
from src.generation.generate_response import generate_response
from src.prompts.rag_prompt import rag_prompt

class RAGPipeline:

    def __init__(self, chunk_type = "fixed",
                 embedding_type="instructor",
                 retrieval_type="hybrid",
                 reranker_type="bge",
                 generation_model="ollama"):
        
        # docs = load_docs("https://docs.langchain.com/oss/python/deepagents/overview")
        self.chunk_type = chunk_type
        self.embedding_type = embedding_type
        self.retrieval_type = retrieval_type
        self.reranker_type = reranker_type
        self.generation_model = generation_model

        print("Initializing the RAG pipeline")

        # docs = load_docs([
        #     "langchain",
        #     "fastapi",
        #     "pytorch",
        #     "tensorrt",
        # ])

        # cleaned_docs = clean_docs(docs)
        # self.chunks = get_chunks(chunk_type, cleaned_docs)

        CACHE_DIR = "data/cache"

        os.makedirs(CACHE_DIR, exist_ok=True)

        chunks_path = f"{CACHE_DIR}/chunks.pkl"

        if os.path.exists(chunks_path):

            print("Loading cached chunks...")

            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            print(f"Total chunks indexed: {len(self.chunks)}")

        else:

            print("Loading docs from source...")

            docs = load_docs([
                "langchain",
                "fastapi",
                "pytorch",
                "tensorrt",
            ])

            print(f"Total raw docs loaded: {len(docs)}")

            cleaned_docs = clean_docs(docs)

            print(f"Total cleaned docs:{len(cleaned_docs)}")


            self.chunks = get_chunks(chunk_type, cleaned_docs)
            print(f"Total chunks indexed: {len(self.chunks)}")

            with open(chunks_path, "wb") as f:
                pickle.dump(self.chunks, f)

            print("Chunks cached successfully")

        # for chunk in self.chunks:
        #     source = chunk.metadata.get("source", "unknown")

        #     text = chunk.page_content[:300]

        #     stable_id = hashlib.md5(
        #         f"{source}_{text}".encode()
        #     ).hexdigest()[:12]

        #     chunk.metadata["doc_id"] = f"{source}_{stable_id}"
        # for idx, chunk in enumerate(self.chunks):
        #     chunk.metadata["doc_id"] = idx 
        for chunk in self.chunks:

            source = chunk.metadata.get("source", "").lower()

            if "langchain" in source:
                prefix = "langchain"

            elif "fastapi" in source:
                prefix = "fastapi"

            elif "pytorch" in source:
                prefix = "pytorch"

            elif "tensorrt" in source:
                prefix = "tensorrt"

            else:
                prefix = "unknown"

            text = chunk.page_content.strip()[:500]

            stable_hash = hashlib.md5(
                f"{prefix}_{text}".encode("utf-8")
            ).hexdigest()[:12]

            chunk.metadata["doc_id"] = f"{prefix}_{stable_hash}"
        
        self.embeddings = get_embeddings(embedding_type)

        # self.vectorstore = faiss_vectorstore(self.chunks, self.embeddings)
        faiss_path = f"data/vectorstores/faiss_{chunk_type}_{embedding_type}"

        self.vectorstore = faiss_vectorstore(
            self.chunks,
            self.embeddings,
            save_path=faiss_path
        )        

        self.retriever = get_retriever(retrieval_type)
        self.reranker = get_reranking(reranker_type)
        self.llm = get_generation_model(generation_model)
        self.retrieval_type = retrieval_type

    def run(self, query, k=5, temperature=0.2, session_history=None):
        if self.retrieval_type == "hybrid":
            docs = self.retriever(
                self.vectorstore,
                self.chunks,
                query
            )
        else:
            docs = self.retriever(
                self.vectorstore,
                query
            )
        reranked_docs = self.reranker.rerank(query, docs, top_k=k)
        context = "\n\n".join(
            [
                doc.page_content
                for doc in reranked_docs
            ]
        )

        chat_history = ""

        if session_history:
            messages = session_history.messages[-6:]
            chat_history = "\n".join([f"{msg.type}:{msg.content}" for msg in messages])

        prompt = rag_prompt(query, context, chat_history=chat_history)
        response = generate_response(self.llm, prompt, temperature)

        return {
            "query":query,
            "response":response,
            "context":context,
            "retrieved_docs" : docs,
            "reranked_docs": reranked_docs
        }
