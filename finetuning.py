"""
Script to fine tune selected model (model_name) with SimCSE implementation from Sentence Transformers library
Recommended to run on GPU
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from sentence_transformers import losses
from sentence_transformers import datasets
from sentence_transformers import InputExample
import transformers

print(transformers.__version__)

fiqa_train = pd.read_csv("./datasets/FiQA/FiQA_train.csv")

# Define sentence transformer model
model_name = 'sentence-transformers/msmarco-distilbert-cos-v5'
word_embedding_model = models.Transformer(model_name, max_seq_length=216)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Convert train sentences to sentence pairs
train_data = []
for _,row in fiqa_train.iterrows():
    train_data.append(InputExample(texts=[row["question"],row["answer"]]))

train_dataloader = datasets.NoDuplicatesDataLoader(train_data, batch_size=8)
# Use the Multiple Negatives Ranking Loss
train_loss = losses.MultipleNegativesRankingLoss(model)

# Call the fit method with 4 spochs
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,
    show_progress_bar=True  
)

model.save('./models/simcse-model')