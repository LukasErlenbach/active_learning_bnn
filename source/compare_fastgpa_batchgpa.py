"""
compare_fastgpa_batchgpa.py

This script compares the Fast GPA Sampler to the Batch GPA Sampler.

To change the experiment setup, change the values of the variables size_pool,
size_test, random_seed below, or edit the config.yaml files in configs/.

Loads configs from the "configs" directory.
Saves results in "results_fgpa_vs_bgpa_rs_" plus the used random seed.
"""


from time import time
import matplotlib.pyplot as plt
import logging

from run_experiment import prepare_experiments, run_experiment
from utils.plotting import plot_rmse_metrics
from utils.timing import secs_to_str

log_level = logging.INFO
size_pool = 0
size_test = 0
random_seed = 0
config_fgpa = "configs/config_small_fgpa.yaml"
config_bgpa = "configs/config_small_bgpa.yaml"

results_dir = "results_fgpa_vs_bgpa_rs_" + str(random_seed) + "/"
ds = prepare_experiments(results_dir, size_pool, size_test, random_seed, log_level)

metrics_fgpa = run_experiment(ds, results_dir, config_fgpa, random_seed)
metrics_bgpa = run_experiment(ds, results_dir, config_bgpa, random_seed)

title = "FastGPA Sampler took " + secs_to_str(metrics_fgpa.time_sampling) + " secs\n"
title += "Batch GPA Sampler took " + secs_to_str(metrics_bgpa.time_sampling) + " secs"
labels = ["FastGPA Sampler", "Batch GPA Sampler"]
fig = plot_rmse_metrics([metrics_fgpa, metrics_bgpa], labels, title=title)

figpath = results_dir + "comparison_fgpa_bgpa"
print("Saving figure as ", figpath)
fig.savefig(figpath)
