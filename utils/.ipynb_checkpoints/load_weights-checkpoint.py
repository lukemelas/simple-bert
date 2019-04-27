"""
Load TensorFlow checkpoints into PyTorch model. 
"""

import numpy as np
import tensorflow as tf
import torch

def load_param(checkpoint_file, conversion_table):
    """
    Load parameters according to conversion_table.
    Args:
        checkpoint_file (string): pretrained checkpoint model file in tensorflow
        conversion_table (dict): { pytorch tensor in a model : checkpoint variable name }
    """
    for pyt_param, tf_param_name in conversion_table.items():
        tf_param = tf.train.load_variable(checkpoint_file, tf_param_name)

        # for weight(kernel), we should do transpose
        if tf_param_name.endswith('kernel'):
            tf_param = np.transpose(tf_param)

        assert pyt_param.size() == tf_param.shape, \
            'Dim Mismatch: %s vs %s ; %s' % \
                (tuple(pyt_param.size()), tf_param.shape, tf_param_name)
        
        # assign pytorch tensor from tensorflow param
        pyt_param.data = torch.from_numpy(tf_param)


def load_transformer(model, checkpoint_file):
    """
    Load transformer, ** not heads ** , into PyTorch model.  
    """

    # Embedding layer
    e, p = model.embed, 'bert/embeddings/'
    load_param(checkpoint_file, {
        e.tok_embed.weight: p+"word_embeddings",
        e.pos_embed.weight: p+"position_embeddings",
        e.seg_embed.weight: p+"token_type_embeddings",
        e.norm.gamma:       p+"LayerNorm/gamma",
        e.norm.beta:        p+"LayerNorm/beta"
    })

    # Transformer blocks
    for i in range(len(model.blocks)):
        b, p = model.blocks[i], "bert/encoder/layer_%d/"%i
        load_param(checkpoint_file, {
            b.attn.proj_q.weight:   p+"attention/self/query/kernel",
            b.attn.proj_q.bias:     p+"attention/self/query/bias",
            b.attn.proj_k.weight:   p+"attention/self/key/kernel",
            b.attn.proj_k.bias:     p+"attention/self/key/bias",
            b.attn.proj_v.weight:   p+"attention/self/value/kernel",
            b.attn.proj_v.bias:     p+"attention/self/value/bias",
            b.proj.weight:          p+"attention/output/dense/kernel",
            b.proj.bias:            p+"attention/output/dense/bias",
            b.pwff.fc1.weight:      p+"intermediate/dense/kernel",
            b.pwff.fc1.bias:        p+"intermediate/dense/bias",
            b.pwff.fc2.weight:      p+"output/dense/kernel",
            b.pwff.fc2.bias:        p+"output/dense/bias",
            b.norm1.gamma:          p+"attention/output/LayerNorm/gamma",
            b.norm1.beta:           p+"attention/output/LayerNorm/beta",
            b.norm2.gamma:          p+"output/LayerNorm/gamma",
            b.norm2.beta:           p+"output/LayerNorm/beta",
        })

def load_weights_for_pretraining(model, checkpoint_file):
    '''
    Load parameters of model for pretraining (i.e. masked LM and sequence classifier)
    from TensorFlow model checkpoint file onto PyTorch model. 
    '''
    
    # Load transformer body
    load_transformer(model.transformer, checkpoint_file)
    
    # Sequence classification (+ pooler) and masked language model (decoder)
    conversion_table = {
        model.pooler.weight:                'bert/pooler/dense/kernel',
        model.pooler.bias:                  'bert/pooler/dense/bias',
        model.seq_relationship.weight:      'cls/seq_relationship/output_weights',
        model.seq_relationship.bias:        'cls/seq_relationship/output_bias',
        model.decoder_linear.weight:        'cls/predictions/transform/dense/kernel',
        model.decoder_linear.bias:          'cls/predictions/transform/dense/bias',
        model.decoder_norm.gamma:           'cls/predictions/transform/LayerNorm/gamma',
        model.decoder_norm.beta:            'cls/predictions/transform/LayerNorm/beta',
        model.decoder_output_bias:          'cls/predictions/output_bias',
    }
    load_param(checkpoint_file, conversion_table)
    
def load_weights_for_classification(model, checkpoint_file):
    '''
    Load parameters of model for classification (i.e. pooler)
    from TensorFlow model checkpoint file onto PyTorch model. 
    '''
    # Load transformer body
    load_transformer(model.transformer, checkpoint_file)
    
    # Sequence classification (+ pooler) and masked language model (decoder)
    conversion_table = {
        model.pooler.weight:                'bert/pooler/dense/kernel',
        model.pooler.bias:                  'bert/pooler/dense/bias',
    }
    load_param(checkpoint_file, conversion_table)
    