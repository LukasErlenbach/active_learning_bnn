"""
dataset.py

This module implements the class Dataset.

The Dataset class holds the data on which the experiments are performed.
It distinguishes between pool, test and training data.
The actual data is only stored *once* and access is granted via reference.
Additional copies with unique access indices can be obtained via .make_copy().
"""

import numpy as np
from sklearn.utils.random import sample_without_replacement


class Dataset:
    # X_pool, y_pool, X_test, y_test are refences to the real data storage
    # due to python's mutable variable paradigm
    # different instances of Dataset hold differnt sets of train/pool_idxs
    def __init__(self, X_pool, y_pool, X_test, y_test, name=None):
        self.train_idxs = np.empty(0)
        self.pool_idxs = np.arange(len(X_pool))
        self._X_pool = X_pool
        self._y_pool = y_pool
        self._X_test = X_test
        self._y_test = y_test
        self.name = name

    def apply_scaler(self, scaler):
        self.scaler = scaler
        self.scaler.fit(self._X_pool)
        self._X_pool = self.scaler.transform(self._X_pool)
        self._X_test = self.scaler.transform(self._X_test)

    def apply_scaler_y(self, scaler):
        self.scaler_y = scaler
        self.scaler_y.fit(self._y_pool)
        self._y_pool = self.scaler_y.transform(self._y_pool)
        self._y_test = self.scaler_y.transform(self._y_test)

    def reset_pool(self):
        self.train_idxs = np.empty(0)
        self.pool_idxs = np.arange(len(self._X_pool))

    # different copies hold reference ot the same data, but different indices
    def make_copy(self, approximate_pool=False):
        ds = Dataset(self._X_pool, self._y_pool, self._X_test, self._y_test, self.name)
        # if training data is not empty -> copy the idxs
        if len(self.train_idxs) > 2:
            ds.train_idxs = np.array(self.train_idxs)
            ds.pool_idxs = np.array(self.pool_idxs)
        return ds

    # subsample data to requested size
    def reduce_size(self, size_pool, size_test):
        assert size_pool <= self._X_pool.shape[0]
        assert size_test <= self._X_test.shape[0]
        pool_sample = sample_without_replacement(
            self._X_pool.shape[0], n_samples=size_pool
        )
        test_sample = sample_without_replacement(
            self._X_test.shape[0], n_samples=size_test
        )
        self._X_pool = self._X_pool[pool_sample]
        self._y_pool = self._y_pool[pool_sample]
        self._X_test = self._X_test[test_sample]
        self._y_test = self._y_test[test_sample]
        self.train_idxs = np.empty(0)
        self.pool_idxs = np.arange(len(self._X_pool))

    def add_to_training(self, idxs, return_data=False):
        if not self.train_idxs.size > 0:
            self.train_idxs = np.array(idxs)
        else:
            assert np.max(idxs) < len(self.pool_idxs)
            self.train_idxs = np.append(self.train_idxs, self.pool_idxs[idxs])

        if return_data:
            added_data = self._X_pool[self.pool_idxs[idxs]]
            self.pool_idxs = np.delete(self.pool_idxs, idxs)
            return added_data
        else:
            self.pool_idxs = np.delete(self.pool_idxs, idxs)

    def X_train(self):
        if not self.train_idxs.size > 0:
            return np.empty(0)
        else:
            return self._X_pool[self.train_idxs]

    def y_train(self):
        if not self.train_idxs.size > 0:
            return np.empty(0)
        else:
            return self._y_pool[self.train_idxs]

    def X_pool(self):
        if not self.pool_idxs.size > 0:
            return np.empty(0)
        else:
            return self._X_pool[self.pool_idxs]

    def y_pool(self):
        if not self.pool_idxs.size > 0:
            return np.empty(0)
        else:
            return self._y_pool[self.pool_idxs]

    def X_test(self):
        return self._X_test

    def y_test(self):
        return self._y_test

    def get_data(self):
        return (
            self.X_train(),
            self.y_train(),
            self.X_pool(),
            self.y_pool(),
            self.X_test(),
            self.y_test(),
        )
