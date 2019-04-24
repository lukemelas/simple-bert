import sys, os
import time, datetime

import numpy as np
import torch

def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]

def eval_bool(x, default=False):
    if x is None:
        return default
    try:
        return bool(eval(x))
    except TypeError:
        return default
