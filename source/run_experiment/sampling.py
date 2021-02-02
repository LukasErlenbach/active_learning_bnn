"""
sampling.property

This module implements various sampling policies.
Each policy hold a reference to X_pool and X_train and implements a
policy.request_samples(, n_samples, learner) function.

Available policies:
    RandomPolicy (samples random points)
    GPAPolicy (Fast GPA Sampler as described in the thesis, based on the
               GPApproximator implementation from gpa.py)
    BatchGPAPolicy (as Fast GPA Sampler, but without fast updating)
"""


import numpy as np
from sklearn.utils.random import sample_without_replacement
from run_experiment.gpa import GPApproximator
from utils import get_global_var


# base class
class Policy:
    def __init__(self):
        pass

    def update(self, X_pool, X_train):
        self.X_pool = X_pool
        self.X_train = X_train


# sample points at random
class RandomPolicy(Policy):
    def __str__(self):
        return "RandomPolicy"

    # returns predicted variance of the network
    def var(self, learner, X_train, X_pool):
        mean, std = learner.predict_w_std(X_pool)
        var = np.power(std, 2)
        return var

    def request_samples(self, n_samples, learner=None):
        idxs_samples = sample_without_replacement(
            self.X_pool.shape[0], n_samples=n_samples
        )
        return idxs_samples


# implementation of the Fast GPA Sampler
class GPAPolicy(Policy):
    def __init__(self, batch_size, n_mc_samples):
        self.batch_size = batch_size
        self.n_mc_samples = n_mc_samples

    def __str__(self):
        return "Fast GPA Policy"

    def var(self, learner, X_train, X_pool):
        # return post variance of pool data
        gpa = GPApproximator(self.n_mc_samples)
        post_var = gpa.eval_post_var(learner, X_train, X_pool)
        return post_var

    def request_samples(self, n_samples, learner):
        logger = get_global_var("logger")
        logger.info(
            "Requesting",
            n_samples,
            "samples from Fast GPA Policy with batch_size ",
            self.batch_size,
        )
        gpa = GPApproximator(self.n_mc_samples)
        # this is the list we want to return
        idxs_sample = []
        # sample at most self.batch_size many points in each iteration
        while len(idxs_sample) < n_samples:
            # standard evaluation
            if len(idxs_sample) == 0:
                post_var = gpa.eval_post_var(learner, self.X_train, self.X_pool)
            # fast updating procedure
            else:
                post_var = gpa.eval_post_var_new_points(max_var_idxs_in_pool)
            # ensure we do not overshoot the requested number of samples
            n_request_max = np.min((self.batch_size, n_samples - len(idxs_sample)))
            # select points with highest post_var values
            max_var_idxs_in_pool = list(post_var.argsort()[-n_request_max:])
            idxs_sample += max_var_idxs_in_pool

        # ensure the sampled idxs are unique
        # -> sampled idxs should have post_var=0
        if len(set(idxs_sample)) < len(idxs_sample):
            raise Exception("GPA Sampler selected some idxs multiple times.")
        return idxs_sample


# implementation of the Batch GPA Sampler
# similar to the Fast GPA Sampler above but without the fast update
class BatchGPAPolicy(Policy):
    def __init__(self, batch_size, n_mc_samples):
        self.batch_size = batch_size
        self.n_mc_samples = n_mc_samples

    def __str__(self):
        return "Batch GPA Policy"

    def request_samples(self, n_samples, learner):
        logger = get_global_var("logger")
        logger.info("Requesting", n_samples, "samples from Batch GPA Policy.")
        gpa = GPApproximator(self.n_mc_samples)
        # this is the list we want to return, real idxs
        idxs_sample = []
        # this is the list we fill during one iteration, reset local idxs
        idxs_sample_iter = []
        # to keep track of the original idxs, use to transform _iter to idxs_sample
        idxs_pool = np.arange(len(self.X_pool))

        counter_all_idxs = lambda: len(idxs_sample) + len(idxs_sample_iter)

        while counter_all_idxs() < n_samples:
            n_request_samples = min(self.batch_size, n_samples - counter_all_idxs())
            # standard evaluation
            post_var = gpa.eval_post_var(learner, self.X_train, self.X_pool)
            max_var_idxs_in_pool = post_var.argsort()[-n_request_samples:]
            if n_request_samples == 1:
                max_var_idxs_in_pool = [max_var_idxs_in_pool]
            idxs_sample_iter = max_var_idxs_in_pool

            if counter_all_idxs() < n_samples:
                # idxs in the original pool
                real_idxs_sample_iter = [int(idxs_pool[i]) for i in idxs_sample_iter]
                idxs_sample = idxs_sample + real_idxs_sample_iter
                # update pool and train data
                idxs_pool = np.setdiff1d(idxs_pool, real_idxs_sample_iter)
                remove_from_pool = self.X_pool[idxs_sample_iter, :]
                if len(idxs_sample_iter) == 1:
                    remove_from_pool = remove_from_pool.reshape(1, -1)
                    self.X_train = np.append(self.X_train, remove_from_pool, axis=0)
                self.X_train = np.append(self.X_train, remove_from_pool, axis=0)
                self.X_pool = np.delete(self.X_pool, idxs_sample_iter, axis=0)
                idxs_sample_iter = []

        # add remaining idxs_samples_iter in last iteration
        idxs_sample_iter = [int(idxs_pool[i]) for i in idxs_sample_iter]
        idxs_sample += idxs_sample_iter
        # ensure the sampled idxs are unique
        # -> sampled idxs should have post_var=0
        if len(set(idxs_sample)) < len(idxs_sample):
            raise Exception("GPA Sampler selected some idxs multiple times.")
        return np.array(idxs_sample)
