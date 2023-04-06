import pickle
import time
import pandas as pd
import sys
from nltk.corpus import stopwords
from typing import List
import string
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import warnings
from neuralNetwork import NeuralNetwork

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

techniques = ["p"]
stop = stopwords.words('english')
tasks = ["stars", "useful", "funny", "cool"]


def main():
    if sys.argv[1] == "p":
        arr = process_args(sys.argv[1:])
        if len(arr) == 4:
            technique, train_file, test_file, model = arr[0], arr[1], arr[2], arr[3]
            train_data = preprocess(train_file)
            test_data = preprocess(test_file)
            train_data = train_data.head(35000)
            test_data = test_data.head(35000)
            ## Experiment 2
            # print(train_data['stars'].value_counts()[1])
            # print(train_data['stars'].value_counts()[2])
            # print(train_data['stars'].value_counts()[3])
            # print(train_data['stars'].value_counts()[4])
            # print(train_data['stars'].value_counts()[5])
            # print("trimming")
            # train_data = train_data[train_data.stars != 1]
            # test_data = test_data[test_data.stars != 1]
        else:
            technique, valid_file, model = arr[0], arr[1], arr[2]
            test_data = preprocess(valid_file)
            test_data = test_data.head(35000)
            train_data = None

        validate_args(technique)
        probabilistic(train_data, test_data, model)
    elif  sys.argv[1] == "n":
        process_args(sys.argv[1:])


def clean_text(text):
    punc = ''.join([char for char in text if char not in string.punctuation])
    stop = [word for word in punc.split() if word.lower() not in stopwords.words('english')]
    ## Comment out line 54 and 57 for experiment 3
    # return punc
    return stop


def probabilistic(train_data, test_data, file):
    models = {}
    start = time.time()
    if file:
        file = open(file, 'rb')
        models, vec = pickle.load(file)
        X_test = vec.transform(test_data['text'])

    else:
        vec = CountVectorizer(analyzer=clean_text)
        X_train = vec.fit_transform(train_data['text'])
        X_test = vec.transform(test_data['text'])
        for task in tasks:
            y_train = train_data[[task]]
            clf = MultinomialNB()
            clf.fit(X_train, y_train)
            models[task] = clf
        filename = "p" + '.sav'
        file = open(filename, 'wb')
        pickle.dump([models, vec], file)
    print(f"Done training in {round(time.time()-start, 2)} seconds ")
    file.close()
    predictions = {}
    for task in tasks:
        clf = models[task]
        y_pred = clf.predict(X_test)
        predictions[task] = y_pred

    for task in tasks:
        y_test = test_data[[task]]
        y_pred = predictions[task]
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"TASK: {task} ||| F1: {round(f1*100,2)} -- ACCURACY: {round(accuracy_score(y_test, y_pred)*100,2)}")



def clean_data(data):
    """ Can be implemented if our models share certain data cleaning methods
        My model uses tokenization, removal of stop words, removal of punctuation
    """
    pass


def validate_args(technique):
    if technique not in techniques:
        sys.exit("Invalid classification method")


def process_args(args) -> List[str]:
    if args[0] == "p":
        if args[-1] != "p.sav":
            args.append(None)
            return [args[0], args[1], args[2], args[3]]
        else:
            return [args[0], args[1], args[2]]
    elif args[0] == "n":
        predictType = args[1]
        modelPath = None
        trainPath = None
        if "json" in args[2]:
            trainPath = args[2]
        else:
            modelPath = args[2]
        testPath = args[3]
        NeuralNetwork(modelPath, predictType, train_set=trainPath, test_set=testPath)




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
