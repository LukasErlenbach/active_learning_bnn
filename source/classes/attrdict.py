"""
attrdict.py

from https://danijar.com/patterns-for-fast-prototyping-with-tensorflow/

Simple dict with faster access to items.
"""


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
