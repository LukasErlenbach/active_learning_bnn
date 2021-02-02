"""
run_experiment.py

This modul represents the current API of the project.

It prepares, runs and performs the actual active learning experiment.

In particular it implements the function prepare_experiments() which calls functions
from the run_experiment/initialize.py and run_experiment/prepare_variables.py modules.
As well as the function run_experiment() which can be utilized to run a single
experiment, i.e. which loads a config.yaml and initializes an independent networks.
"""


from time import time
from utils import set_global_var, get_global_var
from .prepare_variables import prepare_global_vars
from .initialize import (
    init_tf_session,
    init_rs,
    init_dataset,
    get_initial_idxs,
)
from .make_exp_config import make_exp_config


def add_metrics(metric1, metric2):
    for key in metric1:
        metric1[key] += metric2[key]
    return metric1


def perform_active_learning(ex_name, ds, model, sample_policy, al_schedule, seed):
    # set seed again to get the same initial training points for fair comparison
    init_rs(seed)
    logger = get_global_var("logger")
    set_global_var("spolicy", sample_policy)

    train_indices = get_initial_idxs(ds, al_schedule.num_train_init)
    ds.add_to_training(train_indices)
    sample_policy.update(ds.X_pool(), ds.X_train())
    logger.info("initial training on ", len(ds.X_train()), " data points")
    metrics = model.train(ds.X_train(), ds.y_train(), ds.X_test(), ds.y_test())
    metrics.time_sampling = 0.0
    for i in range(al_schedule.al_iter):
        st = time()
        idxs_sample = sample_policy.request_samples(al_schedule.num_al_incr, model)
        time_sampling = time() - st
        ds.add_to_training(idxs_sample)
        logger.info(
            "Adding",
            len(idxs_sample),
            "idxs to training data. Total:",
            len(ds.X_train()),
        )
        # accout for final iteration in training
        start_epoch = metrics.epochs[-1] + model.ts.eval_after_epochs.step
        new_metrics = model.train(
            ds.X_train(),
            ds.y_train(),
            ds.X_test(),
            ds.y_test(),
            start_epoch=start_epoch,
        )
        new_metrics.time_sampling = time_sampling
        metrics = add_metrics(metrics, new_metrics)
        sample_policy.update(ds.X_pool(), ds.X_train())
    return metrics


def prepare_experiments(results_dir, size_pool, size_test, random_seed, log_level):
    init_rs(random_seed)
    prepare_global_vars(results_dir, random_seed, log_level)
    ds = init_dataset("housing", size_pool, size_test)
    return ds


def run_experiment(ds, results_dir, config_yaml, random_seed):
    config = make_exp_config(ds, config_yaml, random_seed, results_dir)

    logger = get_global_var("logger")
    logger.set_prefix(config.name)
    logger.info("\n\nStarting new experiment:", config.name)
    logger.info(config.important_stats())
    metrics = perform_active_learning(
        config.name,
        config.ds,
        config.model,
        config.sample_policy,
        config.al_schedule,
        config.seed,
    )
    return metrics
