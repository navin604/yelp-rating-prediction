import preprocessing
import time

from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset

dataset = "data/yelp_academic_dataset_review.json"
invalid_test_set = "data/test.JSON"
train_set = "data/train_set.json"
test_set = "data/test_set.json"
validation_set = "data/validation_set.json"


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("Preprocessing...")
    train_data = preprocessing.preprocess(validation_set)
    test_data = preprocessing.preprocess(test_set)
    # Load the tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5)

    # Prepare the data
    train_texts = train_data['text'].tolist()
    train_labels = [label - 1 for label in train_data['stars'].tolist()]

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)

    val_texts = test_data['text'].tolist()
    val_labels = [label - 1 for label in test_data['stars'].tolist()]
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = pd.DataFrame(
        {'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'],
         'labels': train_labels})
    val_dataset = pd.DataFrame(
        {'input_ids': val_encodings['input_ids'], 'attention_mask': val_encodings['attention_mask'],
         'labels': val_labels})
    train_dataset = Dataset.from_pandas(train_dataset)
    val_dataset = Dataset.from_pandas(val_dataset)
    print(train_dataset.unique("labels"))
    print(val_dataset.unique("labels"))
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        label_names=['labels']
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    print("Training...")
    trainer.train()
    print("Evaluating...")
    eval_results = trainer.evaluate()

    print(eval_results)


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    finally:
        t1 = time.time()
        print(f"time to complete: {t1-t0}")
