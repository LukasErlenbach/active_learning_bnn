"""
compare_gpa_rand.py

This script compares the performance of the Fast GPA Sampler to a baseline of
randomly selection points in the active learning iterations.

To change the experiment setup, change the values of the variables size_pool,
size_test, random_seed below, or edit the config.yaml files in configs/.

Loads configs from the "configs" directory.
Saves results in "results_gpa_vs_rand_rs_" plus the used random seed.
"""


from time import time
import matplotlib.pyplot as plt
import logging

from run_experiment import prepare_experiments, run_experiment
from utils.plotting import plot_rmse_metrics

log_level = logging.INFO
size_pool = 0
size_test = 0
random_seed = 0
config_gpa = "configs/config_gpa.yaml"
config_rand = "configs/config_rand.yaml"

results_dir = "results_gpa_vs_rand_rs_" + str(random_seed) + "/"
ds = prepare_experiments(results_dir, size_pool, size_test, random_seed, log_level)

metrics_gpa = run_experiment(ds, results_dir, config_gpa, random_seed)
metrics_rand = run_experiment(ds, results_dir, config_rand, random_seed)

title = "FastGPA Sampler vs random point selection"
labels = ["FastGPA Sampler", "Random Selection"]
fig = plot_rmse_metrics([metrics_gpa, metrics_rand], labels, title=title)

figpath = results_dir + "comparison_gpa_rand"
print("Saving figure as ", figpath)
fig.savefig(figpath)
