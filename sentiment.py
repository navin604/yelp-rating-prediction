import pickle
import pandas as pd
import sys
from nltk.corpus import stopwords
from typing import List
import string
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

techniques = ["p"]
stop = stopwords.words('english')
tasks = ["stars", "useful", "funny", "cool"]


def main():
    train_file, test_file, technique, model = process_args(sys.argv[1:])
    validate_args(technique)
    train_data = preprocess(train_file)
    test_data = preprocess(test_file)

    train_data = train_data.head(10000)
    test_data = test_data.head(10000)
    if technique == "p":
        probabilistic(train_data, test_data, model)


def clean_text(text):
    punc = ''.join([char for char in text if char not in string.punctuation])
    stop = [word for word in punc.split() if word.lower() not in stopwords.words('english')]
    return stop


def probabilistic(train_data, test_data, model):
    models = {}
    if model:
        # X_test = X
        # y_test = y
        # file = open(model, 'rb')
        # pca, clf = pickle.load(file)
        pass

    else:
        print("PREPROCESSING DATA")
        vec = CountVectorizer(analyzer=clean_text)
        X_train = vec.fit_transform(train_data['text'])
        X_test = vec.transform(test_data['text'])
        print("TRAINING MODELS")
        for task in tasks:
            print(f"STARTIN{task}")
            y_train = train_data[[task]]
            y_test = test_data[[task]]
            clf = MultinomialNB()
            clf.fit(X_train, y_train)
            models[task] = clf
        filename = "p" + '.sav'
        file = open(filename, 'wb')
        pickle.dump(models, file)
    file.close()

    predictions = {}
    for task in tasks:
        clf = models[task]
        y_pred = clf.predict(X_test)
        predictions[task] = y_pred

    for task in tasks:
        y_test = test_data[[task]]
        y_pred = predictions[task]
        print(classification_report(y_test, y_pred))
        print(accuracy_score(y_test, y_pred)*100)
        # text = res.split('\n')
        # edited_text = [line for line in res if not line.startswith('macro')]
        # new_report = '\n'.join(edited_text)




def clean_data(data):
    """ Can be implemented if our models share certain data cleaning methods
        My model uses tokenization, removal of stop words, removal of punctuation
    """
    pass


def validate_args(technique):
    if technique not in techniques:
        sys.exit("Invalid classification method")


def process_args(args) -> List[str]:
    if len(args) == 3:
        args.append(None)
    return args[0], args[1], args[2], args[3]


def preprocess(file_name) -> pd.DataFrame:
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
            return dataSet
        dataSet = pd.concat([dataSet, holder])


if __name__ == "__main__":
    main()
