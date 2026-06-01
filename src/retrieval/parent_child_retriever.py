from collections import defaultdict

def parent_child_retriver(
        vectorstore, parent_docs, query, k=5):
    
    child_results = vectorstore.similarity_search(query, k=k)
    parent_lookup = {
        doc.metadata["parent_id"]: doc
        for doc in parent_docs
    }

    retrieved_parents = []
    seen = set()

    for child in child_results:
        parent_id = child.metadata["parent_id"]

        if parent_id not in seen:
            retrieved_parents.append(
                parent_lookup[parent_id])
            seen.add(parent_id)
    return retrieved_parents