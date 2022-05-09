"""
Reranks bm25 candidates from previous retrieval step and returns index of final answer prediction
"""

from scipy import spatial

def bert_reranker(query, bm25_candidates, embedding_docs, model):
    """
    Parameters:
        query: User query of string type
        bm25_candidates: List of integers representing answer indexes, from bm25 retrieval
        embedding_docs: list of answers embeddings e.g. [[answer1 embedding], [answer2 embedding]...]
        model: selected model
    Returns:
        pred_index: Integer index of final prediction

    """
    query_embedding = model.encode(query)
    candidates_embeddings = [embedding_docs[candidate_index] for candidate_index in bm25_candidates]
    cossim_dict = {}
    for index, c_embed in enumerate(candidates_embeddings):
        cossim_dict[index] = 1 - float(spatial.distance.cosine(c_embed, query_embedding))
    cossim_dict = {k: v for k, v in sorted(cossim_dict.items(), key=lambda item: item[1], reverse=True)}
    if list(cossim_dict.values())[0] < 0.50:
        return -1
    return bm25_candidates[list(cossim_dict.keys())[0]]