"""
plotter.py

This moduls implements the Plotter class.
Usually an instance of Plotter is initialized as global varianble at the start
of the program flow. A Plotter can visualize metrics as well as models and
policies.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import get_global_var


def moving_avg(data, width):
    avg = pd.Series(data).rolling(width).mean().iloc[width - 1 :].values
    return avg


def plot_metrics(ax, epochs, data, label, avg_w=1, color=None):
    epochs, data = moving_avg(epochs, avg_w), moving_avg(data, avg_w)
    ax.plot(epochs, data, label=label, color=color)


def plot_rmse_metrics(runs_metrics, labels, title, avg_w=10):
    plt.rcParams.update({"font.size": 15})
    colors = ["C" + str(i) for i in range(10)]
    plt.close()
    fig, ax = plt.subplots(2, 1, figsize=(15, 8))
    fig.suptitle(title)
    for i, metrics in enumerate(runs_metrics):
        label = labels[i]
        color = colors[i]
        plot_metrics(
            ax[0],
            metrics.epochs,
            metrics.train_rmses,
            label,
            avg_w=avg_w,
            color=color,
        )
        plot_metrics(
            ax[1],
            metrics.epochs,
            metrics.test_rmses,
            label,
            avg_w=avg_w,
            color=color,
        )
    ax[0].set_ylabel("train RMSE")
    ax[1].set_ylabel("test RMSE")
    for a in ax.flatten():
        a.legend()
        a.set_xlabel("training epochs")
    return fig
