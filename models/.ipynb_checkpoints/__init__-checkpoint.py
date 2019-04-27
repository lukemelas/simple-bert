import os, sys
import torch

from . import transformer, heads

def get_model_for_classification(args):
    ''' Load a model in full or half precision with pretrained weights. '''

    # Load a BERT model
    if args.model == 'bert':
        cfg = transformer.TransformerConfig.from_json(args.cfg)
        body = transformer.Transformer(cfg)
        model = heads.TransformerForClassification(cfg, body, args.num_labels)
    
    # Load pretrained weights
    if args.load_weights:
        if '.pth' in args.load_weights: # PyTorch file
            model.load_state_dict(torch.load(args.load_weights))
        elif '.ckpt' in args.load_weights: # TensorFlow file
            from utils.load_weights import load_weights_for_classification
            load_weights_for_classification(model, args.load_weights)

    # CUDA / half-precision / distributed training
    model = distribute_and_fp16(args, model)    
    return model

def get_model_for_pretrain(args):
    ''' Load a model in full or half precision with pretrained weights. '''

    # Load a BERT model
    if args.model == 'bert':
        cfg = transformer.TransformerConfig.from_json(args.cfg)
        body = transformer.Transformer(cfg)
        model = heads.TransformerForPretrain(cfg, body)
    
    # Load pretrained weights
    if args.load_weights:
        if '.pth' in args.load_weights: # PyTorch file
            model.load_state_dict(torch.load(args.load_weights))
        elif '.ckpt' in args.load_weights: # TensorFlow file
            from utils.load_weights import load_weights_for_classification
            load_weights_for_classification(model, args.load_weights)

    # CUDA / half-precision / distributed training
    model = distribute_and_fp16(args, model)    
    return model

def distribute_and_fp16(args, model):
    ''' Multi-GPU and half-precision '''
    
    if args.fp16:
        model.half()
    model.to(args.device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("To use FP16, install apex from https://www.github.com/nvidia/apex")
        model = DDP(model)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return model