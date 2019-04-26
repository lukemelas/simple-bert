"""
Helper Functions
"""

import os
import random
import logging

import numpy as np
import torch


def set_seeds(seed):
    '''Set all random seeds'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    '''Removes tokens until inputs have the same length'''
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def get_random_word(vocab_words):
    '''Unform random word from vocab'''
    i = random.randint(0, len(vocab_words)-1)
    return vocab_words[i]

def get_logger(name, log_path):
    '''Gets a logger'''
    logger = logging.getLogger(name)
    fomatter = logging.Formatter(
        '[ %(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
    if not os.path.isfile(log_path): f = open(log_path, "w+")
    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(fomatter)
    logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(fomatter)
    logger.addHandler(streamHandler)
    logger.setLevel(logging.DEBUG)
    return logger
