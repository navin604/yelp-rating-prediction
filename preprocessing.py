import pandas as pd


def preprocess(file_name):
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
            return dataSet[0:10000]
        dataSet = pd.concat([dataSet, holder])
