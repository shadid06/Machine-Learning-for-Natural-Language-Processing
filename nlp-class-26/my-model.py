from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./sentiment-model",
    tokenizer="./sentiment-model",
)

reviews = [
    "The product is wonderful!",
    "I am unhappy with this purchase.",
]

results = classifier(reviews)

for review, result in zip(reviews, results):
    print("Review:", review)
    print("Prediction:", result["label"])
    print("Score:", round(result["score"], 4))
    print("-" * 30)
