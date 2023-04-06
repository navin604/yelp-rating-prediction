import nltk
import pickle
import time
import pandas as pd
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import List
import string
from nltk.classify import NaiveBayesClassifier, accuracy
from nltk.stem import PorterStemmer
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
techniques = ["p", "rf"]
stop = stopwords.words('english')
categories = ['stars', 'useful', 'funny', 'cool']


def main():
    if sys.argv[1] == "p":
        arr = process_args(sys.argv[1:])
        if len(arr) == 4:
            technique, train_file, test_file, model = arr[0], arr[1], arr[2], arr[3]
            train_data = preprocess(train_file)
            test_data = preprocess(test_file)
            train_data = train_data.head(35000)
            test_data = test_data.head(35000)
        else:
            technique, valid_file, model = arr[0], arr[1], arr[2]
            test_data = preprocess(valid_file)
            test_data = test_data.head(35000)
            train_data = None

        validate_args(technique)
        probabilistic(train_data, test_data, model)
    elif sys.argv[1] == "n":
        process_args(sys.argv[1:])
    elif sys.argv[1] == "rf":
        process_args(sys.argv[1:])


def probabilistic(data):
    # Tokenize review
    data['text'] = data['text'].apply(word_tokenize)
    # Remove stop words and punctuation
    data['text'] = data['text'].apply(lambda words: ' '.join([word for word in words if word not in stop]))
    data['text'] = data['text'].apply(lambda words: ''.join([word for word in words if word not in string.punctuation]))
    data['text'] = data['text'].apply(lambda words: ''.join([PorterStemmer().stem(word) for word in words]))
    print(data)

    X = data["text"]
    y = data["stars"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    # Vectorization
    vec = TfidfVectorizer()
    X_train_tf = vec.fit_transform(X_train)
    X_test_tf = vec.transform(X_test)

    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(X_train_tf, y_train)
    y_pred = naive_bayes_classifier.predict(X_test_tf)

    # Calculate accuracy
    print(accuracy_score(y_test, y_pred))


def random_trees(train_data, test_data, model, task):
    start = time.time()

    if not model:
        print("No model used. Training new model...")
        X_train = train_data['text']
        X_test = test_data['text']
        if task == "stars":
            y_train = train_data[['stars']]
            y_test = test_data[['stars']]

            m = RandomForestClassifier(criterion="gini", n_jobs=-1)

        else:
            # y_train = train_data[['cool', 'useful', 'funny']]
            y_train = train_data[[task]]
            # y_test = test_data[['cool', 'useful', 'funny']]
            y_test = test_data[[task]]

            m = RandomForestRegressor(n_jobs=-1)

        # Vectorize
        vectorizer = CountVectorizer(ngram_range=(1,1), lowercase=True)

        print("fit_transform vectorizer")
        X_train_v = vectorizer.fit_transform(X_train)
        print("transform vectorizer")
        X_test_v = vectorizer.transform(X_test)

        print("Fitting model...")
        m.fit(X_train_v, y_train)

        with open(f"rf_{task}.sav", "wb") as model_file:
            pickle.dump([vectorizer, m], model_file)

    else:
        print("Using pre-trained model...")
        try:
            # Get validation data
            X_test = test_data['text']
            if task == "stars":
                y_test = test_data[[task]]
                with open(model, "rb") as model_file:
                    vectorizer, m = pickle.load(model_file)
            else:
                y_test = test_data[[task]]
                with open(model, "rb") as model_file:
                    vectorizer, m = pickle.load(model_file)

            X_test_v = vectorizer.transform(X_test)

        except FileNotFoundError:
            print("Model not found. Exiting...")
            return
        except EOFError:
            print("Model is incorrect or is corrupt. Exiting...")
            return

    print("Predicting model...")
    y_pred = m.predict(X_test_v)
    if task == "stars":
        print(classification_report(y_test, y_pred))
    else:
        mae = mean_absolute_error(y_test, y_pred)
        print('MAE:', mae)

        # Compute mean squared error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        print('MSE:', mse)

        # Compute R-squared (R2)
        r2 = r2_score(y_test, y_pred)
        print('R2:', r2)
    print(f"Time: {time.time() - start}")


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
        # NeuralNetwork(modelPath, predictType, train_set=trainPath, test_set=testPath)

    elif args[0] == "rf":
        if len(args) != 4:
            print("Incorrect number of arguments")
            return
        else:
            if args[1] == "stars" or args[1] == "cool" or args[1] == "funny" or args[1] == "useful":
                task_type = args[1]
                if "sav" in args[2]:
                    train_data = None
                    model_path = args[2]
                    test_data = preprocess(args[3])
                    test_data = test_data.head(25000)
                else:
                    model_path = None
                    train_data = preprocess(args[2])
                    test_data = preprocess(args[3])
                    train_data = train_data.head(25000)
                    test_data = test_data.head(25000)
                    # Experiment 3
                    # train_data = train_data[train_data.stars != 5]
                    # test_data = test_data[test_data.stars != 5]
                random_trees(train_data=train_data, test_data=test_data, model=model_path, task=task_type)
            else:
                print("Incorrect task type")
                return


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
