"""
global_variables.py

When this module gets loaded, a dict() of global variables is initialized.
This is really handy for using a global logger, plotter and random seed yet usage
of global variables should be avoided whenever possible.

Implements set_global_var() and get_global_var() for definition and access
of global variables.
"""

GLOBAL_VARIABLES = dict()


def set_global_var(var_name, var_value):
    global GLOBAL_VARIABLES
    GLOBAL_VARIABLES[var_name] = var_value
    return var_value


def get_global_var(var_name):
    global GLOBAL_VARIABLES
    if not var_name in GLOBAL_VARIABLES.keys():
        msg = "Global variable " + str(var_name) + " was not set."
        raise Exception(msg)
    return GLOBAL_VARIABLES[var_name]
