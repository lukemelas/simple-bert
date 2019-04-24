import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class LightConvLanguageModel(nn.Module):
    '''A class that essentially wraps the decoder and provides utils.'''

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, src_tokens, src_lengths):
        """
        Run the forward pass for a decoder-only model.
        Feeds a batch of tokens through the decoder to predict the next tokens.
        Args:
            src_tokens (LongTensor): tokens on which to condition the decoder,
                of shape `(batch, tgt_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
        Returns:
            the decoder's output, typically of shape `(batch, seq_len, vocab)`
        """
        return self.decoder(src_tokens)

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

        # # NOTE: Not sure what is going on here <-- figure this out
        # if not hasattr(args, 'max_source_positions'):
        #     args.max_source_positions = args.tokens_per_sample
        # if not hasattr(args, 'max_target_positions'):
        #     args.max_target_positions = args.tokens_per_sample

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # For language modeling, size of input and outpur are the same
        args.decoder_input_dim = args.decoder_output_dim = args.decoder_embed_dim

        # Simple embedding
        embed_tokens = nn.Embedding(len(task.dictionary), args.decoder_input_dim, padding_idx=task.dictionary.pad())
        # NOTE: task.dictionary.pad needs to be replaced

        # Convolutional decoder
        decoder = LightConvDecoder(
            args=args, 
            dictionary=task.output_dictionary, 
            embed_tokens=embed_tokens
        )

        return LightConvLanguageModel(decoder)



class LightConvDecoder(nn.Module):
    """
    LightConvDecoder decoder consisting of *args.decoder_layers* layers, each
    of which is a :class:`LightConvDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``True``
        final_norm (bool, optional): whether to apply layer norm in the final layer
            Default: ``False``
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=True, final_norm=False):
        super().__init__(dictionary)
        
        # Embeddings
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed
        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim
        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        
        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            LightConvDecoderLayer(args, no_encoder_attn, kernel_size=args.decoder_kernel_size_list[i])
            for i in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, output_embed_dim, bias=False)
            if embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
            )
            inner_states.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]
