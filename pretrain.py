import os, sys, argparse
import time, datetime
import random
import logging

import numpy as np
import torch
from torch import nn

from train.pretrain_data import PipelineForPretrain, SentencePairDataLoader
from train.optim import get_optimizer
from train.pretrain_trainer import Trainer
from models import get_model_for_pretrain
from utils import utils, tokenization

# Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--text_file", 
                    default=None, type=str, required=True,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--model", 
                    default=None, type=str, required=True,
                    help="Model name, for example: 'bert' ")
parser.add_argument("--cfg", 
                    default=None, type=str, required=True,
                    help="Model configuration file")
parser.add_argument("--exp_name",
                    default=None, type=str, required=True,
                    help="Experiment output directory")

# Other parameters
parser.add_argument("--load_weights", 
                    type=str, default=None, 
                    help="A .ckpt or .pth file with a pretrained model.")
parser.add_argument("--max_seq_length",
                    type=int, default=512,
                    help="Sequences longer than this will be truncated.")
parser.add_argument("--val_every",
                    type=int, default=-1,
                    help="Validate on dev set every [X] iterations. Default is -1 (never).")
parser.add_argument("--checkpoint_every",
                    type=int, default=1000,
                    help="Save model checkpoint every [X] iterations.")
parser.add_argument("--do_lower_case",
                    action='store_true',
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--vocab", 
                    type=str, default='config/bert-uncased-vocab.txt', 
                    help="File containing a BERT vocabulary.")
parser.add_argument("--train_batch_size",
                    type=int, default=32,
                    help="Total batch size for training.")
parser.add_argument("--val_batch_size",
                    type=int, default=8,
                    help="Validation/test batch size.")
parser.add_argument("--learning_rate",
                    type=float, default=2e-5,
                    help="The initial learning rate for Adam.")
parser.add_argument("--total_iterations",
                    type=float, default=100000, # 1000000
                    help="Total number of training iterations to perform.")
parser.add_argument("--warmup_proportion",
                    type=float, default=0.1,
                    help="Linear training warmup proportion.")
parser.add_argument("--local_rank",
                    default=-1, type=int,
                    help="For distributed training on gpus")
parser.add_argument('--seed',
                    default=42, type=int,
                    help="Random seed")
parser.add_argument('--gradient_accumulation_steps',
                    default=1, type=int,
                    help="Number of updates steps to accumulate before backprop.")
parser.add_argument('--fp16',
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale',
                    type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                            "0 (default value): dynamic loss scaling.\n"
                            "Positive power of 2: static loss scaling value.\n")
parser.add_argument("--no_cuda", action='store_true',
                    help="Disable GPUs and run on CPU.")
parser.add_argument('--no_tensorboard', action='store_true',
                    help="Disable tensorboard")

# Parse and check args
start_time = time.time()
args = parser.parse_args()

# Create output directory for saving models and logs
args.output_dir = os.path.join('experiments', 'pretrain', args.exp_name)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
logger = utils.get_tensorboard_logger(args)

# Select device
if args.local_rank == -1 or args.no_cuda:
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    args.n_gpu = 1

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')

# Log GPU information
logger.add_text('info', f"args: {args}")

# Modify batch size if accumulating gradients
args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps 

# Reproducibility
utils.set_seeds(args.seed, multi_gpu=args.n_gpu > 0)

# Build dataloaders 
tokenizer = tokenization.FullTokenizer(args.vocab, do_lower_case=args.do_lower_case)
tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))
pipeline = [PipelineForPretrain(max_pred=20, # what is this?
                                mask_prob=0.15, # actually this does nothing
                                vocab_words=list(tokenizer.vocab.keys()), # 
                                indexer=tokenizer.convert_tokens_to_ids,
                                max_len=args.max_seq_length)]
dataloader = SentencePairDataLoader(args.text_file,
                                    batch_size=args.train_batch_size,
                                    tokenize=tokenize,
                                    max_len=args.max_seq_length,
                                    pipeline=pipeline)

# Model, optimizer
model = get_model_for_pretrain(args)
optimizer = get_optimizer(args, model, t_total=1000000)

# Train
epoch = 0
trainer = Trainer(logger)
while trainer.global_step < args.total_iterations:

    # # Validation is not yet implemented -- run it for as long as possible
    # if args.val > 0 and trainer.global_step % args.val_every == 0:
    #     trainer.evaluate(args, model, val_dataloader, criterion) # TODO: add val

    # Train for one epoch
    trainer.train(args, model, dataloader, optimizer, epoch)
    epoch += 1

# Save trained model
trainer.save(args, model)
logger.info('Done pretraining!')
