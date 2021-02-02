"""
timing.py

This module implements time related functions.
"""

from datetime import timedelta


def secs_to_str(secs):
    return str(timedelta(seconds=round(secs)))
