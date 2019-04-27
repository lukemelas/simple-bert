"""
Helper Functions
"""

import os
import random
import numpy as np
import torch

def set_seeds(seed, multi_gpu=True):
    '''Set all random seeds'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if multi_gpu:
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

def get_tensorboard_logger(args):
    '''Gets a TensorBoard logger or creates a fallback'''
    print(f"Logging to {args.output_dir}")
    try: 
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(log_dir=args.output_dir) # this crashes my VM 
        print(f"Connect with: \n\t tensorboard --logdir {args.output_dir} --port 6001")
    except:
        print('NOTE: TensorBoardX is not installed. Logging to console.')
        class NotSummaryWriter(object): pass;
        logger = NotSummaryWriter()
        nothing_function = lambda s, *args, **kw: None
        logger.add_text = logger.add_scalar = nothing_function
    logger.info = print # could also write to a file here but this is fine for now
    return logger
