"""
initialize.py

This module gets called from run_experiment.py and initializes various variables.
In particular: global random seed, tensorflow session, the network model, the
sampling policy, the dataset and the initial training points.
"""

import numpy as np

import tensorflow.compat.v1 as tf

from sklearn.preprocessing import StandardScaler
from sklearn.utils.random import sample_without_replacement

from models import BNN
from utils.global_variables import set_global_var, get_global_var
from datasets import (
    get_housing_dataset_pickle,
)

from run_experiment.sampling import RandomPolicy, GPAPolicy, BatchGPAPolicy

# set all needed random seeds
def init_rs(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)


# each model needs and holds its own tf.Session() instance
def init_tf_session():
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    return sess


# initialize a network model with tensorflow session
# models are can be found in source/models/
# raises: Exception if name_model is unknown
def init_model(name_model, train_schedule, net_config, seed):
    tf.reset_default_graph()
    init_rs(seed)
    # only bnn available here
    if name_model == "bnn":
        model = BNN(net_config=net_config, train_schedule=train_schedule)
    else:
        raise Exception(
            "Unknown model requested: " + str(policy_name) + " (valid input: {bnn})"
        )
    sess = init_tf_session()
    model.set_session(sess)
    return model


# initialize a sample policy
# sample policies are stored in source/sampling.py
# raises: Exception if policy_name is unknown
def init_sample_policy(policy_name, sampling_batch_size, n_mc_samples):
    if policy_name == "random":
        return RandomPolicy()
    elif policy_name == "gpa":
        return GPAPolicy(sampling_batch_size, n_mc_samples)
    elif policy_name == "bgpa":
        return BatchGPAPolicy(sampling_batch_size, n_mc_samples)
    else:
        raise Exception(
            "Unknown sampling policy requested: "
            + str(policy_name)
            + " (valud: {random, gpa, bgpa})"
        )


# initialize a dataset, resizes it and applies a StandardScaler
def init_dataset(name_dataset, size_pool, size_test):
    logger = get_global_var("logger")

    # only housing data is available here
    assert name_dataset == "housing"

    # load data
    ds = get_housing_dataset_pickle()

    # size_pool and size_test are 0? -> do not resize dataset
    if size_pool != 0 or size_test != 0:
        ds.reduce_size(size_pool, size_test)

    scaler = StandardScaler(copy=False)
    ds.apply_scaler(scaler)
    scaler_y = StandardScaler(copy=False)
    ds.apply_scaler_y(scaler_y)

    logger.info(
        "Working on dataset:",
        ds.name,
        "with pool data shape:",
        ds.X_pool().shape,
        "and test data shape:",
        ds.X_test().shape,
    )
    logger.info("Applying scaler:", str(scaler))
    return ds


# returns the initial training idxs
def get_initial_idxs(ds, n_points):
    return sample_without_replacement(ds.X_pool().shape[0], n_points)
