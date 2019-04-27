import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# GELU Activation: https://arxiv.org/abs/1606.08415
gelu = lambda x : x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# LayerNorm
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
except ImportError:
    class LayerNorm(nn.Module):
        "Layer normalization in the TF style (epsilon inside the square root)."
        def __init__(self, cfg, variance_epsilon=1e-12):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(cfg.dim))
            self.beta  = nn.Parameter(torch.zeros(cfg.dim))
            self.variance_epsilon = variance_epsilon

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.gamma * x + self.beta

def Linear(in_features, out_features, bias=True):
    ''' Wrapper for nn.Linear '''
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    ''' Wrapper for nn.Embedding '''
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

class Embeddings(nn.Module):
    '''Embedding with optional position and segment type embeddings.'''
    def __init__(self, cfg, position_embeds=True, segment_embeds=True):
        super().__init__()
        self.tok_embed = Embedding(cfg.vocab_size, cfg.dim) # token embedding
        self.pos_embed = Embedding(cfg.max_len, cfg.dim) if position_embeds else None # position embedding
        self.seg_embed = Embedding(cfg.n_segments, cfg.dim) if segment_embeds else None # segment(token type) embedding
        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg):
        e = self.tok_embed(x)
        if self.pos_embed is not None:
            pos = torch.arange(x.size(1), dtype=torch.long, device=x.device) # x.size(1) = seq_len
            pos = pos.unsqueeze(0).expand_as(x) # (S,) -> (B, S)
            e = e + self.pos_embed(pos)
        if self.seg_embed is not None:
            e = e + self.seg_embed(seg) 
        return self.drop(self.norm(e))

