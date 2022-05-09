"""
Get top 100 candidate answers given user query as part of retrieval step.
Okapi BM25 with default parameters used. 
"""

import numpy as np

def bm25_score(query, docs, top_n=100 , k=1.2, b=0.75):
    """
    Parameters:
        query: User query of string type
        docs: list of answers e.g. [[first answer], [second answer]]
        top_n: number of results to return
        k, b: default parmeters of bm25 without optimization
    
    Returns:
        bm25_results: list of document index of top_n results as per bm25 score

    """
    avg_doc_len = sum(len(doc) for doc in docs)/len(docs)
    query_tokens = query.split()
    bm25_results = {}
    nq_dict = {}
    for query_token in query_tokens:
        nq = sum([1 for doc in docs if query_token in doc])
        nq_dict[query_token] = nq
    for index,doc in enumerate(docs):
        doc_len = len(doc)
        doc_bm25 = 0
        for query_token in query_tokens: 
            # Calculating BM25 TF
            freq = doc.count(query_token)
            tf = (freq*(k+1)) / (freq + (k * (1-b+(b*(doc_len/avg_doc_len)))))
            # Calculating IDF
            nq = nq_dict[query_token]
            idf = np.log((len(docs) - nq + 0.5) / (nq + 0.5)) + 1
            doc_bm25 += tf*idf
        bm25_results[index] = doc_bm25
    bm25_results = {k: v for k, v in sorted(bm25_results.items(), key=lambda item: item[1],reverse=True)}
    return list({k: bm25_results[k] for k in list(bm25_results)[:top_n]}.keys( ))



