"""
gpa.py

This module implements a GPApproximator which can be used to compute the
posterior variance of an approximate Gaussian Process.
"""

import numpy as np

# init defines number of Monte Carlo samples used for the estimation and the
# regularization la(mbda)
class GPApproximator:
    def __init__(self, n_samples_mc=25, la=0.01):
        self.n_samples_mc = n_samples_mc
        self.la = la
        # member variables set during computation
        self.samples = None
        self.cov_est = None
        self.post_cov_est = None

    # expects the learner to implement a .sample_y function
    # samples self.n_samples_mc many instances and centralizes them
    def sample_y_values(self, learner, X_data):
        self.samples = np.array(learner.sample_y(X_data, self.n_samples_mc))
        self.samples -= np.mean(self.samples, axis=0)
        assert self.samples.shape[0] == self.n_samples_mc

    # estimate covariance matrix and saves it to self.cov_est
    # if resample==True, call self.sample_y_values()
    def estimate_cov(self, learner, X_data, resample):
        if self.samples is None or resample:
            self.sample_y_values(learner, X_data)
        self.cov_est = self.samples.T.dot(self.samples) / self.n_samples_mc
        return self.cov_est

    # calls self.estimate_cov() and return diagonal of self.cov_est
    def eval_sample_var(self, learner, X_anchor, X_pool, estimate_cov=True):
        if estimate_cov:
            self.estimate_cov(learner, np.append(X_anchor, X_pool, axis=0), True)
        return np.diag(self.cov_est)[len(X_anchor) :]

    # use self.cov_est to estimate posterior variance
    # sets self.post_cov_est
    def eval_post_var(self, learner, X_anchor, X_pool, estimate_cov=True):
        len_X_anchor = len(X_anchor)
        len_X_pool = len(X_pool)
        # calculate self.cov_est
        if estimate_cov:
            self.estimate_cov(learner, np.append(X_anchor, X_pool, axis=0), True)
        # self.cov_est is (len(X_anchor) + len(X_pool))**2
        idxs_anchor = np.arange(len_X_anchor)
        idxs_pool = np.arange(len_X_anchor, len_X_anchor + len_X_pool)
        assert (self.cov_est == self.cov_est.T).all()

        # get submatrices
        cov_old_old = self.cov_est[np.ix_(idxs_anchor, idxs_anchor)]
        cov_old_new = self.cov_est[np.ix_(idxs_anchor, idxs_pool)]
        cov_new_new = self.cov_est[np.ix_(idxs_pool, idxs_pool)]

        # add self.la*identidy and invert
        coo_inv = np.linalg.inv(cov_old_old + np.eye(len(idxs_anchor)) * self.la)
        # calculate posterior covariance matrix
        self.post_cov_est = cov_new_new - cov_old_new.T.dot(coo_inv).dot(cov_old_new)
        return np.diag(self.post_cov_est)

    # fast updating procedure, calculates new posterior covariance matrix,
    # after idxs_in_X_pool have been selected
    def eval_post_var_new_points(self, idxs_in_X_pool):
        if len(idxs_in_X_pool) == 1:
            return self.eval_post_var_new_point(idxs_in_X_pool[0])
        selected = (
            self.post_cov_est[np.ix_(idxs_in_X_pool, idxs_in_X_pool)]
            + np.eye(len(idxs_in_X_pool)) * self.la
        )
        covariances = self.post_cov_est[idxs_in_X_pool, :]
        update_mat = covariances.T.dot(np.linalg.inv(selected)).dot(covariances)
        self.post_cov_est -= update_mat
        # ensure that selected idxs have zero covariance
        self.post_cov_est[idxs_in_X_pool, :] = 0
        self.post_cov_est[:, idxs_in_X_pool] = 0
        return np.diag(self.post_cov_est)

    # same as above, optimized for len(idx_in_X_pool)==1
    def eval_post_var_new_point(self, idx_in_X_pool):
        submat = self.post_cov_est[idx_in_X_pool, :].reshape(-1, 1)
        update_mat = submat.dot(submat.T)

        update_mat = update_mat / (
            self.post_cov_est[idx_in_X_pool, idx_in_X_pool] + self.la
        )
        self.post_cov_est = self.post_cov_est - update_mat
        self.post_cov_est[idx_in_X_pool, :] = 0
        self.post_cov_est[:, idx_in_X_pool] = 0
        return np.diag(self.post_cov_est)
