import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common_layers import LayerNorm, gelu

class TransformerForPretraining(nn.Module):
    """A model in the style of BERT for pretraining. 
       Example (1): 
       ```
       transformer = models.Transformer(cfg)
       bert = models.TransformerForPretraining(cfg, transformer)
       ```
       Example (2):
       ```
       lwdc_transformer = models.LightweightTransformer(cfg)
       bert = models.TransformerForPretraining(cfg, lwdc_transformer)
       ```
    """
    def __init__(self, cfg, transformer):
        super().__init__()
        self.transformer = transformer
        
        # Final fully connected layer (BertPooler)
        self.dense = nn.Linear(cfg.dim, cfg.dim)
        self.activation = nn.Tanh()

        # For masked LM 
        self.linear = nn.Linear(cfg.dim, cfg.dim)
        self.norm = LayerNorm(cfg)

        # For sentence classification
        self.classifier = nn.Linear(cfg.dim, 2)
        
        # Tie weights
        embed_weight = self.transformer.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activation(self.dense(h[:, 0]))
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(gelu(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        logits_clsf = self.classifier(pooled_h)
        return logits_lm, logits_clsf

