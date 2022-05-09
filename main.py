print()
print ("FinQABot> Hey! Give me a minute to finish my coffee!")

from bm25_retrieval import bm25_score
from preprocessing import clean_text
from bert_reranker import bert_reranker
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

print("...")

# Loading prerequisites
processed_docs = pickle.load(open("./datasets/FiQA/processed_docs.pkl", 'rb'))
raw_docs = pickle.load(open("./datasets/FiQA/raw_docs.pkl", 'rb'))
embedding_docs = pickle.load(open("./embeddings/finetuned_answers_embeddings.pkl","rb"))
model = SentenceTransformer("./models/simcse-model")
print("FinQABot> Done!(Type 'exit' to exit)")
print()
while True:
    query = input("FinQABot> What is your question?\nYou> ")
    if query=="exit":
        print("See you next time!")
        break
    else:
        bm25_candidates = bm25_score(clean_text(query),processed_docs)
        pred_index = bert_reranker(query, bm25_candidates, embedding_docs, model)
        if pred_index == -1:
            print("FinQABot> Sorry I did not understand your question!")
        else:
            print(f"FinQABot> {raw_docs[pred_index]}")
        print()