# নিজের Dataset দিয়ে NLP Model Fine-tuning

# python -m venv .venv
# .venv\Scripts\activate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score
from datasets import Dataset
import numpy as np
texts = [
    "I love this product.",
    "The quality is excellent.",
    "Very happy with my purchase.",
    "The product works perfectly.",
    "Amazing quality and great service.",
    "I would recommend this product.",
    "Delivery was fast and easy.",
    "This is worth the money.",
    "The product is fantastic.",
    "I am very satisfied.",
    "I hate this product.",
    "The quality is terrible.",
    "Very disappointed with my purchase.",
    "The product stopped working.",
    "The service was awful.",
    "I would not recommend this.",
    "Delivery was very slow.",
    "This is a waste of money.",
    "The product is defective.",
    "I am extremely dissatisfied."
]

labels = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]


# 1. Our labeled dataset
texts = [
    "I love this product.",
    "The quality is excellent.",
    "Very happy with my purchase.",
    "The product works perfectly.",
    "Amazing quality and great service.",
    "I would recommend this product.",
    "Delivery was fast and easy.",
    "This is worth the money.",
    "The product is fantastic.",
    "I am very satisfied.",
    "I hate this product.",
    "The quality is terrible.",
    "Very disappointed with my purchase.",
    "The product stopped working.",
    "The service was awful.",
    "I would not recommend this.",
    "Delivery was very slow.",
    "This is a waste of money.",
    "The product is defective.",
    "I am extremely dissatisfied.",
]

# 1 = Positive, 0 = Negative
labels = [1] * 10 + [0] * 10

dataset = Dataset.from_dict({
    "text": texts,
    "labels": labels,
})

# 2. Split data: training and evaluation
splits = dataset.train_test_split(
    test_size=0.25,
    seed=42,
)

# 3. Load pretrained tokenizer
model_name = "distilbert/distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
    )


tokenized = splits.map(
    tokenize_batch,
    batched=True,
)

# 4. Load pretrained model for 2 classes
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)

# 5. Pad batches to the required length
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

# 6. Measure accuracy on evaluation data


def compute_metrics(eval_pred):
    logits, true_labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(
            true_labels,
            predictions,
        )
    }


# 7. Configure training
training_args = TrainingArguments(
    output_dir="./sentiment-results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=2,
    report_to="none",
    save_total_limit=1,
)

# 8. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)

# 9. Fine-tune the model
trainer.train()

# 10. Evaluate the model
metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)

# 11. Save the fine-tuned model and tokenizer
trainer.save_model("./sentiment-model")
tokenizer.save_pretrained("./sentiment-model")

print("Fine-tuned model saved successfully!")
