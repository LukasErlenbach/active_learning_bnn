"""
yaml_load_dump.py

This module implements all interactions with .yaml files and is based on ruamel.

We store the experiment configurations in .yaml files and the functions in this
module load and dump the configs in a persistent way for reproducablity.

If variables of configs are not set by the .yaml file, defaults are loaded from
source/utils/default_configs.py.
"""


import ruamel.yaml as yaml
from classes.attrdict import AttrDict
from utils.default_configs import default_net_config, default_train_schedule


def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.load(file, Loader=yaml.Loader)
        for key in data.keys():
            if isinstance(data[key], dict):
                data[key] = AttrDict(data[key])
    return AttrDict(data)


def dump_yaml(file_path, data):
    with open(file_path, "w") as file:
        yaml.dump(data.copy(), file, default_flow_style=False)


def update_dict(d_to_update, d_with_update):
    for key in d_with_update:
        d_to_update[key] = d_with_update[key]
    return d_to_update


def load_complete_config_yaml(file_path, num_inputs):
    config = load_yaml(file_path)

    parms = config.parms
    name_model = parms.name_model
    al_schedule = config.al_schedule

    nc_from_file = config.net_config
    nc = default_net_config(name_model, num_inputs)
    nc = update_dict(nc, nc_from_file)

    ts_from_file = config.train_schedule
    ts_from_file.eval_after_epochs = range(
        0, ts_from_file.num_epochs, ts_from_file.eval_after_epochs
    )
    ts_from_file.log_after_epochs = range(
        0, ts_from_file.num_epochs, ts_from_file.log_after_epochs
    )
    ts = default_train_schedule(name_model)
    ts = update_dict(ts, ts_from_file)

    seed = config.seed
    return parms, al_schedule, nc, ts, seed


def dump_complete_config_yaml(
    file_path, parms, al_schedule, net_config, train_schedule, seed
):
    dump_dict = dict()
    dump_dict["parms"] = dict(parms)
    dump_dict["al_schedule"] = dict(al_schedule)
    dump_dict["net_config"] = dict(net_config)
    dump_dict["net_config"].pop("clip_gradient")
    dump_dict["net_config"].pop("num_inputs")
    dump_dict["net_config"].pop("weight_std")
    dump_dict["train_schedule"] = dict(train_schedule)
    dump_dict["train_schedule"]["eval_after_epochs"] = dump_dict["train_schedule"][
        "eval_after_epochs"
    ].step
    dump_dict["train_schedule"]["log_after_epochs"] = dump_dict["train_schedule"][
        "log_after_epochs"
    ].step
    dump_dict["seed"] = seed
    dump_yaml(file_path, dump_dict)
