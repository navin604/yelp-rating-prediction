import time
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error, r2_score
from datasets import Dataset

dataset = "data/yelp_academic_dataset_review.json"
invalid_test_set = "data/test.JSON"
train_set = "data/train_set.json"
test_set = "data/test_set.json"
validation_set = "data/validation_set.json"


def compute_metrics_for_regression(eval_pred):
    preds, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    single_squared_errors = ((preds - labels).flatten() ** 2).tolist()

    # Compute accuracy
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)

    return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    microprecision, microrecall, microf1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'macro f1': f1,
        'macro precision': precision,
        'macro recall': recall,
        'micro f1': microf1,
        'micro precision': microprecision,
        'micro recall': microrecall,
    }


def preprocess(file_name, sizeDataReturned=10000):
    json_reader = pd.read_json(file_name, orient='records', lines=True, chunksize=1000000)
    dataSet = next(json_reader)
    while True:
        holder = next(json_reader, None)
        if holder is None:
            # drops all records missing 'text', 'stars', 'useful', 'funny', or 'cool'
            dataSet = (dataSet.loc[:, ['text', 'stars', 'useful', 'funny', 'cool']].dropna())
            # drops all records without any text, stars value not between 1 and 5, and negative values in optional field
            dataSet = dataSet.loc[(dataSet['text'].str.len() > 0) & (dataSet['stars'].between(1, 5)) & ~(
                    dataSet.loc[:, ['useful', 'funny', 'cool']] < 0).any(axis=1)]
            # setting this to only 10000 records in interest of time
            return dataSet[0:sizeDataReturned]
        dataSet = pd.concat([dataSet, holder])


class NeuralNetwork:
    # from google.colab import drive
    # drive.mount('/content/drive')
    def __init__(self, fromSave, predictType, train_set=None, test_set=None):
        torch.cuda.empty_cache()
        self.train_set = train_set
        self.test_set = test_set
        self.fromSave = fromSave
        self.predictType = predictType
        # train_set = "/content/drive/MyDrive/datasets/train_set_mini.json"
        # test_set = "/content/drive/MyDrive/datasets/test_set.json"
        # fromSave = "/content/drive/MyDrive/datasets/model.sav"
        fromSave = None
        # predictType = "funny"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        print("Preprocessing...")
        if self.fromSave:
            self.evaluateModel()
        else:
            self.trainModel()

    def evaluateModel(self):
        test_data = preprocess(self.test_set, 10000)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        nLabels = 5 if self.predictType == "stars" else 1
        model = RobertaForSequenceClassification.from_pretrained(self.fromSave, num_labels=nLabels)
        test_texts = test_data['text'].tolist()
        test_encodings = tokenizer(test_texts, truncation=True, padding=True)
        if self.predictType == "stars":
            test_labels = [label - 1 for label in test_data['stars'].tolist()]
        else:
            test_labels = [label for label in test_data[self.predictType].tolist()]
        test_dataset = pd.DataFrame(
            {'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask'],
             'labels': test_labels})
        test_dataset = Dataset.from_pandas(test_dataset)
        trainer = Trainer(
            model=model,
            compute_metrics=compute_metrics,
            eval_dataset=test_dataset
        )
        if self.predictType != "stars":
            trainer.compute_metrics = compute_metrics_for_regression
        res = trainer.evaluate()
        print(f"res: {res}")

    def trainModel(self):
        train_data = preprocess(self.train_set, 10000)
        test_data = preprocess(self.test_set, 10000)
        # load tokenizer/model
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        modelName = self.fromSave if self.fromSave else 'roberta-base'
        nLabels = 5 if self.predictType == "stars" else 1
        model = RobertaForSequenceClassification.from_pretrained(modelName, num_labels=nLabels)
        # prepare data
        train_texts = train_data['text'].tolist()
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_texts = test_data['text'].tolist()
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)
        if self.predictType == "stars":
            train_labels = [label - 1 for label in train_data['stars'].tolist()]
            val_labels = [label - 1 for label in test_data['stars'].tolist()]
        else:
            val_labels = [float(label) for label in test_data[self.predictType].tolist()]
            train_labels = [float(label) for label in train_data[self.predictType].tolist()]
        train_dataset = pd.DataFrame(
            {'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'],
             'labels': train_labels})
        val_dataset = pd.DataFrame(
            {'input_ids': val_encodings['input_ids'], 'attention_mask': val_encodings['attention_mask'],
             'labels': val_labels})
        train_dataset = Dataset.from_pandas(train_dataset)
        val_dataset = Dataset.from_pandas(val_dataset)
        # setup trainer
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy='epoch',
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            label_names=['labels'],
            learning_rate=.001
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        if self.predictType != "stars":
            trainer.compute_metrics = compute_metrics_for_regression
        print("Training...")
        trainer.train()
        # print("Evaluating...")
        # eval_results = trainer.evaluate()
        # print(eval_results)
        modelName = "nn_" + self.predictType + ".sav"
        trainer.save_model("/models/" + modelName)
