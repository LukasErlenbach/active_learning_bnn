"""
default_configs.py

This module contains defaults for al_schedule, train_schedule and net_config.
It is called by source/utils/yaml_load_dump.py.
"""


from classes.attrdict import AttrDict


def default_al_schedule():
    config = AttrDict()
    config.num_train_init = 100
    config.al_iter = 4
    config.num_al_incr = 50
    return config


def default_train_schedule(name_model, num_epochs=1000):
    config = AttrDict()
    config.num_epochs = num_epochs
    config.eval_after_epochs = range(0, config.num_epochs, 250)
    config.log_after_epochs = range(0, config.num_epochs, 500)
    config.batch_size = 64
    config.temperature = 0.5
    config.has_uncertainty = True
    return config


def default_net_config(name_model, num_inputs=1):
    config = AttrDict()
    config.num_inputs = num_inputs
    config.layer_sizes = [50, 50]
    config.divergence_scale = 1.0
    config.learning_rate = 3e-3
    config.weight_std = 0.1
    config.clip_gradient = 100.0
    config.dropout_rate = 0.0
    return config
