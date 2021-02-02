"""
base_model.py

This module implements the BaseModel class from which the other network models
are derived. The class holds a tensorflow session and a net_config.

The class implements training, evaluation and prediction functions.
"""


from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import tensorflow.compat.v1 as tf

import numpy as np
import scipy.stats

from classes.attrdict import AttrDict
from utils import get_global_var


class BaseModel:
    def __init__(self, net_config):
        self.net_config = net_config
        self.session = None

    def set_session(self, sess):
        self.session = sess

    """
    This is the core training function for all neural networks.
    Training is performed in the classical SGD fashion on random subsets of the
    training data. A AttrDict with metrics is returned after training.
    """

    def train(self, X_train, y_train, X_test, y_test, start_epoch=0, n_epochs=None):
        if n_epochs is None:
            n_epochs = self.ts.num_epochs
        logger = get_global_var("logger")
        if self.session is None:
            raise Exception("TF session in the network model is not set.")
        if len(X_train) == 0:
            raise Exception("Training data is empty.")

        metrics = AttrDict(
            train_likelihoods=[],
            train_rmses=[],
            test_likelihoods=[],
            test_rmses=[],
            epochs=[],
        )
        # reset optimizer
        self.session.run(self.graph.reset_optimizer)
        # classical training iterations on random subsets of the training data
        # training is performed by calling self.graph.optimize
        for epoch in range(n_epochs):
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            for from_idx in range(0, len(X_train), self.ts.batch_size):
                to_idx = min(from_idx + self.ts.batch_size, len(X_train))
                current = indices[from_idx:to_idx]
                self.session.run(
                    self.graph.optimize,
                    {
                        self.graph.inputs: X_train[current],
                        self.graph.targets: y_train[current],
                        self.graph.num_samples: len(X_train),
                    },
                )
            # eval metrics after defined epochs
            if epoch in self.ts.eval_after_epochs:
                train_llh, train_rmse = self.evaluate(X_train, y_train)
                test_llh, test_rmse = self.evaluate(X_test, y_test)
                metrics.train_likelihoods.append(train_llh)
                metrics.train_rmses.append(train_rmse)
                metrics.test_likelihoods.append(test_llh)
                metrics.test_rmses.append(test_rmse)
                metrics.epochs.append(start_epoch + epoch)

            # print metrics after defined epochs
            if epoch in self.ts.log_after_epochs:
                logger.info(
                    " Epoch:",
                    metrics.epochs[-1],
                    "\n   train log likelihood:",
                    metrics.train_likelihoods[-1],
                    "\n   train rmses:",
                    metrics.train_rmses[-1],
                    "\n   test log likelihood:",
                    metrics.test_likelihoods[-1],
                    "\n   test rmses:",
                    metrics.test_rmses[-1],
                )
        return metrics

    # returns likelihood and rmse
    def evaluate(self, X_data, y_data, batch_size=64):
        likelihoods, squared_distances = [], []
        for index in range(0, len(X_data), batch_size):
            target = y_data[index : index + batch_size]
            feed_dict = {self.graph.inputs: X_data[index : index + batch_size]}

            mean, noise, uncertainty = self.session.run(
                [
                    self.graph.data_mean,
                    self.graph.data_noise,
                    self.graph.data_uncertainty,
                ],
                feed_dict,
            )
            squared_distances.append((target - mean) ** 2)
            if self.has_uncertainty:
                std = np.sqrt(noise ** 2 + uncertainty ** 2 + 1e-8)
            else:
                std = noise
            likelihood = scipy.stats.norm(mean, std).logpdf(target)
            likelihoods.append(likelihood)
        likelihood = np.concatenate(likelihoods, 0).sum(1).mean(0)
        rmse = np.sqrt(np.concatenate(squared_distances, 0).sum(1).mean(0))
        return likelihood, rmse

    # predict y-values for X_data
    def predict(self, X_data, batch_size=64):
        assert len(X_data) > 0
        means = []
        for index in range(0, len(X_data), batch_size):
            feed_dict = {self.graph.inputs: X_data[index : index + batch_size]}

            mean = self.session.run(
                [
                    self.graph.data_mean,
                ],
                feed_dict=feed_dict,
            )
            means.append(mean)
        mean = np.concatenate(means, axis=0)
        return mean

    # predict y-values for X_data and return std as well
    def predict_w_std(self, X_data, batch_size=64):
        assert len(X_data) > 0
        means, noises, uncertainties = [], [], []
        for index in range(0, len(X_data), batch_size):
            feed_dict = {self.graph.inputs: X_data[index : index + batch_size]}
            mean, noise, uncertainty = self.session.run(
                [
                    self.graph.data_mean,
                    self.graph.data_noise,
                    self.graph.data_uncertainty,
                ],
                feed_dict=feed_dict,
            )
            means.append(mean)
            noises.append(noise)
            uncertainties.append(uncertainty)

        mean = np.concatenate(means, axis=0)
        noise = np.concatenate(noises, axis=0)
        if self.has_uncertainty:
            uncertainty = np.concatenate(uncertainties, axis=0)
        std = np.sqrt(noise ** 2 + uncertainty ** 2) if self.has_uncertainty else noise
        return mean, std
