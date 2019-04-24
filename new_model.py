import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common_layers import Linear, LayerNorm, Embedding, gelu

# # TODO: Restructure files:
# - models
#   - common_layers.py
#   - lightweight.py
#   - transformer.py
#   - evolved_transformer.py
#   - heads.py

class LightweightConfig(NamedTuple):
    "Configuration for LWDC model"
    vocab_size: int = None # Size of Vocabulary
    dim: int = 512 # 768 # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 6 # 12 # Numher of Hidden Layers
    n_heads: int = 8 # 12 # Numher of Heads in Multi-Headed Attention Layers
    dim_ff: int = 512*4 # 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    p_drop_conv: float = 0.0 # 0.1 # Probability of Dropout of Attention Layers
    max_len: int = 128 # 512 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of Sentence Segments
    kernel_list: list = [3, 7, 15, 31, 31, 31] # convolutional kernels
    conv_type: str = 'lightweight' # either 'lightweight' or 'dynamic'
    glu_in_conv: bool = True # include generalized linear unit in the conv layer
    norm_before_conv: bool = True # layer norm before conv and after conv, not just after
    weight_softmax: bool = True # softmax the convolutional layer weights
    # tie_weights: bool = True # Share input and output weights # no choice 

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))

    @classmethod
    def check(cfg):
        assert len(cfg.kernel_list) in [1, n_heads]
        assert cfg.conv_type in ['lightweight', 'dynamic']
        assert cfg.dim % n_heads == 0

class LightweightConv(nn.Module):
    '''Lightweight convolution from fairseq.
    Args:
        input_size: # of channels of the input and output
        kernel_size: convolution channels
        padding: padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_softmax: normalize the weight with softmax before the convolution
        dropout: dropout probability
    Forward:
        Input: BxCxT, i.e. (batch_size, input_size, timesteps)
        Output: BxCxT, i.e. (batch_size, input_size, timesteps)
    Attributes:
        weight: learnable weights of shape `(num_heads, 1, kernel_size)`
        bias:   learnable bias of shape `(input_size)`
    '''
    def __init__(self, input_size, kernel_size=1, padding=0, n_heads=1,
                 weight_softmax=True, bias=False, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.n_heads = n_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.Tensor(n_heads, 1, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(input_size)) if bias else None
        self.dropout = dropout
        self.reset_parameters()

    def forward(self, input):
        '''Takes input (B x C x T) to output (B x C x T)'''
        
        # Prepare weight (take softmax)
        B, C, T = input.size()
        H = self.n_heads
        weight = F.softmax(self.weight, dim=-1) if self.weight_softmax else self.weight
        weight = F.dropout(weight, self.weight_dropout, training=self.training)
        
        # Merge every C/H entries into the batch dimension (C = self.input_size)
        # B x C x T -> (B * C/H) x H x T
        # One can also expand the weight to C x 1 x K by a factor of C/H
        # and do not reshape the input instead, which is slow though
        input = input.view(-1, H, T)
        output = F.conv1d(input, weight, padding=self.padding, groups=H)
        output = output.view(B, C, T)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        return output

class ConvBlock(nn.Module):
    """Lightweight or dynamic convolutional layer"""
    def __init__(self, cfg, kernel_size):
        self.norm_before_conv = cgf.norm_before_conv
        
        # Initial fully connected layer or GLU
        self.linear_1 = Linear(cfg.dim, cfg.dim * (2 if cfg.glu_in_conv else 1))
        self.glu = nn.GLU() if cfg.glu_in_conv else None

        # Lightweight or dynamic convolution
        assert cfg.conv_type in ['lightweight', 'dynamic']
        Conv = LightweightConv if cfg.conv_type == 'lightweight' else DynamicConv
        self.conv = Conv(cfg.dim, kernel_size=kernel_size, padding_l=kernel_size-1, # amount of padding
                         weight_softmax=cfg.weight_softmax, n_heads=n_heads, dropout=cfg.p_drop_conv)

        # I do not think this second linear layer is necessary, but we will do it anyway
        self.linear_2 = nn.Linear(cfg.dim, cfg.dim)

        # Dropout and layer normalization
        self.dropout = nn.Dropout(cfg.p_drop_hidden)
        self.conv_layer_norm = LayerNorm(cfg.dim)
        
        # NOTE: This is where the encoder attention would go if there were any

        # Final linear layer: See Figure 2 in the LWDC paper
        self.fc1 = Linear(cfg.dim, cgf.dim_ff)
        self.fc2 = Linear(cgf.dim_ff, cfg.dim)
        self.final_layer_norm = LayerNorm(cfg.dim)

    def __forward__(self, cfg):
        '''See Figure 2(b) in the paper'''
        
        # Linear and GLU
        res = x
        if self.norm_before_conv: 
            x = self.conv_layer_norm(x)
        x = self.dropout(self.linear_1(x))
        x = x if self.glu is None else self.glu(x)

        # Conv
        x = self.conv(x)
        # x = self.linear_2(x) # I don't think this makes sense here
        x = self.dropout(x) # F.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        x = self.conv_layer_norm(x)

        # Linear
        res = x
        if self.norm_before_conv: 
            x = self.final_layer_norm(x)
        x = self.dropout(F.relu(self.fc1(x))) # use gelu?
        x = res + x
        x = self.final_layer_norm(x)
        return x

class LightweightTransformer(nn.Module):
    """A LWDC model in the style of a transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg, position_embeds=False, segment_embeds=True)
        kernel_list = cfg.kernel_list if len(cfg.kernel_list) > 1 else cfg.kernel_list * cfg.n_layers
        # ConvLayer = LightweightConvLayer if cfg.conv_type == 'lightweight' else DynamicConvLayer
        self.blocks = nn.ModuleList([ConvBlock(cfg, kernel_size=k) for k in kernel_list])

    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        for block in self.blocks:
            h = block(h, mask)
        return h