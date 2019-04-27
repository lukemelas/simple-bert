import os, sys, argparse
import time, datetime
import random
import logging

import numpy as np
import torch
from torch import nn

from train import classification_data as data
from train.optim import get_optimizer
from train.classification_trainer import Trainer
from models import get_model_for_classification
from utils import utils, tokenization

# Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", 
                    default=None, type=str, required=True,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--model", 
                    default=None, type=str, required=True,
                    help="Model name, for example: 'bert' ")
parser.add_argument("--cfg", 
                    default=None, type=str, required=True,
                    help="Model configuration file")
parser.add_argument("--task_name",
                    default=None, type=str, required=True,
                    help="The name of the task to train.")
parser.add_argument("--exp_name",
                    default=None, type=str, required=True,
                    help="The name of the experiment output directory where the model predictions and checkpoints will be written.")

# Other parameters
parser.add_argument("--load_weights", 
                    type=str, default=None, 
                    help="A .ckpt or .pth file with a pretrained model.")
parser.add_argument("--max_seq_length",
                    type=int, default=128,
                    help="Sequences longer than this will be truncated.")
parser.add_argument("--val_every",
                    type=int, default=-1,
                    help="Validate on dev set every [X] epochs while training. Default is -1 (never).")
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
parser.add_argument("--num_train_epochs",
                    type=float, default=3.0,
                    help="Total number of training epochs to perform.")
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
parser.add_argument('--do_test', action='store_true',
                    help="[DeepMoji only for now] Evaluate on test after training")

# Parse and check args
start_time = time.time()
args = parser.parse_args()

# Create output directory for saving models and logs
args.task_name = args.task_name.lower()
args.output_dir = os.path.join('experiments', args.task_name, args.exp_name)
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

# Loss function
args.output_mode = data.output_modes[args.task_name]               
if args.output_mode == 'classification': 
    args.output_dtype = torch.long
    criterion = nn.CrossEntropyLoss()
elif args.output_mode == "regression": 
    args.output_dtype = torch.float
    criterion = nn.MSELoss()

# Get the number of labels in our data in order to build the model
args.label_list = data.processors[args.task_name]().get_labels()
args.num_labels = len(args.label_list)

# Build tokenizer
tokenizer = tokenization.FullTokenizer(args.vocab, do_lower_case=args.do_lower_case)

# Build model
model = get_model_for_classification(args)

# Build dataloaders
if args.do_test:
    train_dataloader, val_dataloader, test_dataloader = data.prepare_dataloader(args, tokenizer, test=True)
else:
    train_dataloader, val_dataloader = data.prepare_dataloader(args, tokenizer, test=False)
logger.info(f"***** Loaded train [{len(train_dataloader)}] and val data  [{len(val_dataloader)}]*****")

# Build optimizer
optimizer = get_optimizer(args, model, train_dataloader)

# Trainer
trainer = Trainer(logger)
                    
# Train
for epoch in range(int(args.num_train_epochs)):

    # Validation
    if args.val_every > 0 and epoch % args.val_every == 0:
        trainer.evaluate(args, model, val_dataloader, criterion)

    # Train for one epoch
    trainer.train(args, model, train_dataloader, criterion, optimizer, epoch)

# Save trained model
trainer.save(args, model)

# Finally, evaluate the model again 
loss, result = trainer.evaluate(args, model, val_dataloader, criterion)

# If test 
if args.do_test:
    logger.info('******* Test evaluation *******')
    logger.info("... is not yet implemented")

