def precision_at_k(retrived_ids, relevant_ids, k):

    retrieved_k = retrived_ids[:k]

    relevant_retrieved = len(set(retrieved_k).intersection(set(relevant_ids)))

    return relevant_retrieved/k 

def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved_k = retrieved_ids[:k]

    relevant_retrieved = len(
        set(retrieved_k).intersection(set(relevant_ids))
    )

    return relevant_retrieved /len(relevant_ids)

def hit_rate(retrieved_ids, relevant_ids, k):
    retrieved_k = retrieved_ids[:k]
    hit = any(doc_id in relevant_ids for doc_id in retrieved_k)
    return int(hit)