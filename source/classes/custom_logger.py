"""
custom_logger.py

This module implements the CustomLogger class.

This is a customized version of the standard logging.Logger with additional
output features. At the beginning of the program flow, a CustomLogger gets
initialized as global variable which then can be called from everywhere in
the code.
"""

import logging
from time import strftime, mktime
from utils import secs_to_str


def log(*args):
    return " ".join(map(str, args))


class CustomLogger(logging.Logger):
    def set_prefix(self, prefix):
        self.prefix = prefix

    def info(self, *args):
        super().info(log(self.prefix, " \n ", *args))

    def debug(self, *args):
        super().debug(log(self.prefix, " \n ", *args))

    def warning(self, *args):
        super().warning(log(self.prefix, " \n ", *args))

    def critical(self, *args):
        super().critical(log(self.prefix, " \n ", *args))

    def format_elapsed_time(self, start_time, end_time):
        return (
            "Start time:"
            + strftime("%Y-%m-%d %H:%M:%S", start_time)
            + "\n  End time:"
            + strftime("%Y-%m-%d %H:%M:%S", end_time)
            + "\n  Elapsed time(s):"
            + secs_to_str(mktime(end_time) - mktime(start_time))
        )

    def add_filehandler(self, logpath):
        # clear and create logfile
        open(logpath, "w").close()
        fh = logging.FileHandler(logpath)
        fh.setLevel(self.level)
        formatter = logging.Formatter("%(levelname)s - %(asctime)s - %(message)s")
        fh.setFormatter(formatter)
        self.addHandler(fh)

    def clear_filehandler(self):
        # only keep StreamHandler (which is always initialized first)
        self.handlers = [self.handlers[0]]


def init_logger(log_level):
    logger = CustomLogger("dummy_name")
    logger.set_prefix("")
    logger.setLevel(log_level)
    # create formatter
    formatter = logging.Formatter("%(levelname)s - %(asctime)s - %(message)s")
    # create console handler and set log level
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    # add formatter to console handler
    ch.setFormatter(formatter)
    # add console handler to logger
    logger.addHandler(ch)
    return logger
