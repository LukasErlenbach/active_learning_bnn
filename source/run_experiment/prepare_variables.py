"""
prepare_variables.py

The module gets called from run_experiments.py and initializes several global
valiables such as logger, random seed and the result directory.

If the result directory is not empty, the user is asked in the command line, if
results should be overwritten.
"""

import os
from classes import ExpConfig, init_logger
import tensorflow.compat.v1 as tf
import logging

# tf.disable_v2_behavior()
from utils import set_global_var, get_global_var

# if results_dir does not exists -> create it
# if results_dir does exists -> ask user if it can be overwritten
def prepare_results_dir(results_dir, logger):
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    else:
        response = ""
        while not response in ["yes", "no"]:
            response = input(
                'WARNING Do you want to overwrite the directory "'
                + results_dir
                + '" ? (yes/no) '
            )
        if response == "no":
            raise Exception("Result directory will not be overwritten.")
        else:
            logger.warning("Overwriting the directory ", results_dir)


def prepare_global_vars(results_dir, random_seed, log_level=logging.INFO):
    logger = set_global_var("logger", init_logger(log_level))
    # create dir for saving results
    prepare_results_dir(results_dir, logger)

    # add filehandler after preparing the results directory
    logger.clear_filehandler()
    logger.add_filehandler(results_dir + "logfile")

    infos = ""
    infos += "The current working directory is " + os.getcwd() + "\n"
    infos += "  Saving results in directory " + results_dir + "\n"
    infos += "  Tensorflow version is " + tf.__version__
    logger.info(infos)

    set_global_var("results_dir", results_dir)
    set_global_var("global_seed", random_seed)
    return results_dir
