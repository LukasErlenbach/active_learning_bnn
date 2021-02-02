"""
housing_dataset.py

This module tries to load the housing dataset from 'data/housing.pickle' as a
instance of class Dataset (source/classes/dataset.py).

If called as __main__, the script downloades the dataset from sklearn and creates
a 'housing.pickle' file.

Additional information can be found at:
https://scikit-learn.org/stable/datasets/index.html#california-housing-dataset
"""


import sklearn.datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle

from classes import Dataset


# combine Bedrooms and general rooms, drop geographic infos
def get_housing_dataset():
    data_bunch = sklearn.datasets.fetch_california_housing()
    df = pd.DataFrame(data_bunch["data"], columns=data_bunch["feature_names"])
    df["AveRooms"] += df["AveBedrms"]
    df = df.drop(labels=["Latitude", "Longitude", "AveBedrms"], axis=1)
    X_data = df.values
    y_data = data_bunch["target"][:, np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=0
    )
    return X_train, y_train, X_test, y_test


def get_housing_dataset_pickle():
    try:
        with open("data/housing.pickle", "rb") as file:
            d = pickle.load(file)
        X_train = d["X_train"]
        X_train = X_train
        y_train = d["y_train"]
        X_test = d["X_test"]
        y_test = d["y_test"]
        return Dataset(X_train, y_train, X_test, y_test, "Housing Dataset")

    except Exception as e:
        print(e)
        raise Exception(
            "data/housing.pickle does not exist, please call source/datasets/housing_dataset.py"
        )


def main():
    X_train, y_train, X_test, y_test = get_housing_dataset()

    d = dict()
    d["X_train"] = X_train
    d["y_train"] = y_train
    d["X_test"] = X_test
    d["y_test"] = y_test

    with open("data/housing.pickle", "wb") as file:
        pickle.dump(d, file)
    print("Dumped housing data as data/housing.pickle")

    ds = get_housing_dataset_pickle()
    print("Loaded succesfully from housing.pickle")


if __name__ == "__main__":
    main()
