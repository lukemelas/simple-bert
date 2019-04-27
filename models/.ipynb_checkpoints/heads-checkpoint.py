import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_layers import LayerNorm, gelu

class TransformerForPretrain(nn.Module):
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
        
        # For sentence classification
        self.pooler = nn.Linear(cfg.dim, cfg.dim)
        self.pooler_activation = nn.Tanh()
        self.seq_relationship = nn.Linear(cfg.dim, 2)

        # For masked LM 
        embed_weight = self.transformer.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder_linear = nn.Linear(cfg.dim, cfg.dim)
        self.decoder_norm = LayerNorm(cfg)
        self.decoder_output = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder_output_bias = nn.Parameter(torch.zeros(n_vocab))

        # Tie weights
        self.decoder_output.weight = self.transformer.embed.tok_embed.weight
        
    def forward(self, input_ids, segment_ids=None, input_mask=None, masked_pos=None):
        
        # Allow for null inputs
        segment_ids = torch.zeros_like(input_ids) if segment_ids is None else segment_ids
        input_mask = torch.ones_like(input_ids) if input_mask is None else input_mask
        
        # Transformer
        h = self.transformer(input_ids, segment_ids, input_mask)
        
        # For sentence classification
        pooled_h = self.pooler_activation(self.pooler(h[:, 0]))
        logits_clsf = self.seq_relationship(pooled_h)

        # For masked LM # NOTE: be careful about this masked_pos stuff
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = h if masked_pos is None else torch.gather(h, 1, masked_pos)
        h_masked = self.decoder_norm(gelu(self.decoder_linear(h_masked)))
        logits_lm = self.decoder_output(h_masked) + self.decoder_output_bias
        return logits_lm, logits_clsf

class TransformerForClassification(nn.Module):
    """A model in the style of BERT for classification. 
       Example: 
       ```
       transformer = models.Transformer(cfg)
       bert_for_mrpc_finetuning = models.TransformerForPretraining(cfg, transformer, 7)
       ```
    """
    def __init__(self, cfg, transformer, num_classes):
        super().__init__()
        self.transformer = transformer
        
        # Pooling --> Dropout --> Linear
        self.pooler = nn.Linear(cfg.dim, cfg.dim)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, num_classes)
        
    def forward(self, input_ids, segment_ids=None, input_mask=None, masked_pos=None):
        
        # Allow for null inputs
        segment_ids = torch.zeros_like(input_ids) if segment_ids is None else segment_ids
        input_mask = torch.ones_like(input_ids) if input_mask is None else input_mask
        
        # Transformer --> Pooler --> Dropout --> Linear
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.pooler_activation(self.pooler(h[:, 0]))
        pooled_h = self.dropout(pooled_h)
        logits = self.classifier(pooled_h)
        return logits
