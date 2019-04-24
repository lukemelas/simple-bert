import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class LightConvLanguageModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        '''LightConv and DynamicConv arguments''' 

        # General
        parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', default=0., type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', default=0., type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--input-dropout', type=float, metavar='D',
                            help='dropout probability of the inputs')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', # default=512,
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N', # default=2048,
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N', # default=6,
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N', # default=8, # 16 for Google Billion Words
                            help='num decoder attention heads or LightConv/DynamicConv heads')
        parser.add_argument('--decoder-normalize-before', default=False, action='store_true', # default=True
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', default=False, action='store_true', # default=True,
                            help='share decoder input and output embeddings')

        # LWDC-specific 
        parser.add_argument('--decoder-kernel-size-list', # default=[3, 7, 15, 31, 31, 31],
                            type=lambda x: eval_str_list(x, int), 
                            help='list of kernel size (default: "[3,7,15,31,31,31]")')
        parser.add_argument('--decoder-glu', type=eval_bool,
                            help='glu after in proj')
        parser.add_argument('--decoder-conv-type', default='dynamic', type=str,
                            choices=['dynamic', 'lightweight'],
                            help='type of convolution')
        parser.add_argument('--weight-softmax', default=True, type=eval_bool)
        parser.add_argument('--weight-dropout', type=float, metavar='D',
                            help='dropout probability for conv weights')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # Not sure what is going on here <-- figure this out
        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = args.tokens_per_sample
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = args.tokens_per_sample
        
        embed_tokens = Embedding(len(task.dictionary), args.decoder_input_dim, task.dictionary.pad())

        decoder = LightConvDecoder(args, task.output_dictionary, embed_tokens, no_encoder_attn=True, final_norm=False)

        return LightConvLanguageModel(decoder)
