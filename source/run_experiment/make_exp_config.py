"""
make_exp_config.py

This module is called by run_experiments.py and contains a function, which builds
a ExpConfig (source/classes/exp_config.py) from parameters.
"""


import utils.yaml_load_dump as ydl
from classes import ExpConfig
from run_experiment.initialize import init_model, init_sample_policy


def make_exp_config(ds, exp_conf_path, global_seed, fp_dump_to):
    num_inputs = ds.X_pool().shape[1]

    (
        parms,
        al_schedule,
        net_config,
        train_schedule,
        exp_seed,
    ) = ydl.load_complete_config_yaml(exp_conf_path, num_inputs)

    train_schedule.has_uncertainty = not (
        parms["name_model"] == "det" or parms["name_model"] == "dets"
    )
    # assert that we have sufficient datapoints for the active learning schedule
    if (
        len(ds.X_pool())
        < al_schedule.num_train_init + al_schedule.num_al_incr * al_schedule.al_iter
    ):
        raise Exception("Pool is to small for active learning schedule.")

    model = init_model(
        parms["name_model"], train_schedule, net_config, global_seed + exp_seed
    )
    sample_policy = init_sample_policy(
        parms["sample_policy"], parms["sampling_batch_size"], parms["n_mc_samples"]
    )

    # filename as exp name
    exp_name = exp_conf_path.split("/")[-1]
    # additionally dump config to results folder
    ydl.dump_complete_config_yaml(
        fp_dump_to + exp_name,
        parms,
        al_schedule,
        net_config,
        train_schedule,
        exp_seed,
    )
    return ExpConfig(
        exp_name,
        ds.make_copy(),
        model,
        train_schedule,
        al_schedule,
        sample_policy,
        global_seed + exp_seed,
    )
