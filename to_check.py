import torch
from transformers import pipeline

import clslib


# # Load the fine-tuned RoBERTa-large model (uses GPU automatically if available)
# sentiment_task = pipeline(
#     "sentiment-analysis", 
#     model="siebert/sentiment-roberta-large-english",
#     device_map="auto"
# )

# texts = [
#     "The cinematography was breathtaking, though the pacing felt a bit slow.",
#     "I waited two hours for my order and it arrived completely cold."
# ]

# results = sentiment_task(texts)

# for text, res in zip(texts, results):
#     print(f"Label: {res['label']} 