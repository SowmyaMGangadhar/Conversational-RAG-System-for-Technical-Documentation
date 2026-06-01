from sentence_transformers import CrossEncoder

class BGEReranker:

    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, docs, top_k=5):
        pairs = [(
            query, doc.page_content) for doc in docs
        ]

        scores = self.model.predict(pairs)

        scored_docs = list(zip(docs, scores))
        scored_docs = sorted(
            scored_docs,
            key = lambda x: x[1],
            reverse=True
        )

        reranked_docs = [
            doc 
            for doc, score in scored_docs[:top_k]
        ]

        return reranked_docs