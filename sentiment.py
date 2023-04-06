import pandas as pd
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import string

from nltk.classify import NaiveBayesClassifier, accuracy
from nltk.stem import PorterStemmer
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

techniques = ["p"]
stop = stopwords.words('english')
categories = ['stars', 'useful', 'funny', 'cool']



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
    # Tokenize review
    # data['text'] = data['text'].apply(word_tokenize)
    # Remove stop words and punctuation
    # data['text'] = data['text'].apply(lambda words: ''.join([word for word in words if word not in stop]))
    # data['text'] = data['text'].apply(lambda words: ''.join([word for word in words if word not in string.punctuation]))
    # data['text'] = data['text'].apply(lambda words: ''.join([PorterStemmer().stem(word) for word in words]))
    # Vectorization
    # vec = TfidfVectorizer(analyzer=clean_text)
    # X_train_tf = vec.fit_transform(X_train)
    # X_test_tf = vec.transform(X_test)

    if model:
        # X_test = X
        # y_test = y
        # file = open(model, 'rb')
        # pca, clf = pickle.load(file)
        pass

    else:
        vec = CountVectorizer(analyzer=clean_text)
        X_train = vec.fit_transform(train_data['text'])
        y_train = train_data[['stars']]
        X_test = vec.transform(test_data['text'])
        y_test = test_data[['stars']]
        # # Divide data into training and testing
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        # pca = PCA()
        # X_train = pca.fit_transform(X_train)
        # # Instantiate SVC model
        # clf = make_pipeline(StandardScaler(), LinearSVC(multi_class="ovr", dual=False))
        # clf.fit(X_train, y_train)
        # filename = "svm_" + task + '.sav'
        # file = open(filename, 'wb')
        # pickle.dump([pca, clf], file)
    naive_bayes_classifier = MultinomialNB()
    # naive_bayes_classifier.fit(X_train_tf, y_train)
    naive_bayes_classifier.fit(X_train, y_train)
   # y_pred = naive_bayes_classifier.predict(X_test_tf)
    y_pred = naive_bayes_classifier.predict(X_test)

    #accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))



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
