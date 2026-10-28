from transformers import pipeline

# Load the model that we trained in Class 26
classifier = pipeline(
    "text-classification",
    model="./sentiment-model",
    tokenizer="./sentiment-model",
)


def predict_sentiment(text):
    result = classifier(text)[0]

    # Class 26 dataset:
    # 0 = Negative, 1 = Positive
    label_mapping = {
        "LABEL_0": "Negative",
        "LABEL_1": "Positive",
    }

    raw_label = result["label"]
    sentiment = label_mapping.get(raw_label, raw_label)

    return {
        "text": text,
        "sentiment": sentiment,
        "score": round(result["score"], 4),
    }


reviews = [
    "The delivery was terrible.",
    "I love the quality of this product.",
    "The product arrived today.",
]

for review in reviews:
    prediction = predict_sentiment(review)

    print("Review:", prediction["text"])
    print("Sentiment:", prediction["sentiment"])
    print("Score:", prediction["score"])
    print("-" * 40)
