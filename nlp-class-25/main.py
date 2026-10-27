
from transformers import pipeline

# Load a pre-trained sentiment analysis model
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

texts = [
    "I love learning machine learning.",
    "This product is terrible.",
    "The service was excellent."
]

# Run inference
results = classifier(texts)

for text, result in zip(texts, results):
    print("Text:", text)
    print("Label:", result["label"])
    print("Score:", round(result["score"], 4))
    print("-" * 40)
