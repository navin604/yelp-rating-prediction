import pandas as pd
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import string
from nltk.classify import NaiveBayesClassifier, accuracy
from nltk.stem import PorterStemmer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
techniques = ["p"]
stop = stopwords.words('english')
categories = ['stars', 'useful', 'funny', 'cool']

def main():
    file, technique, model = process_args(sys.argv[1:])
    validate_args(technique)
    data = preprocess(file)

    data = data.head(15000)
    if technique == "p":
        probabilistic(data)


def probabilistic(data):
    # Tokenize review
    data['text'] = data['text'].apply(word_tokenize)
    # Remove stop words and punctuation
    data['text'] = data['text'].apply(lambda words: ' '.join([word for word in words if word not in stop]))
    data['text'] = data['text'].apply(lambda words: ''.join([word for word in words if word not in string.punctuation]))
    data['text'] = data['text'].apply(lambda words: ''.join([PorterStemmer().stem(word) for word in words]))
    print(data)

    X = data[['text']]
    y = data[["stars"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    # Vectorization
    vec = TfidfVectorizer()
    X_train_tfidf = vec.fit_transform(X_train['text'])

    # Transform test data
    X_test_tfidf = vec.transform(X_test['text'])



    nb = NaiveBayesClassifier.train(X_train_tfidf)

    # Make predictions on test data
    predictions = nb.classify_many(X_test_tfidf)

    # Calculate accuracy
    print(accuracy(nb, zip(predictions, y_test['stars'].tolist())))


def clean_data(data):
    """ Can be implemented if our models share certain data cleaning methods
        My model uses tokenization, removal of stop words, removal of punctuation
    """
    pass


def validate_args(technique):
    if technique not in techniques:
        sys.exit("Invalid classification method")


def process_args(args) -> List[str]:
    if len(args) == 2:
        args.append(None)
    return args[0], args[1], args[2],


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
