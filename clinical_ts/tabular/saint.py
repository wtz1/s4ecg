__all__ = ['SAINT', 'SAINTConfig']

from .saint_modules.pretrainmodel import SAINT
from .saint_modules.augmentations import *

import dataclasses
from dataclasses import dataclass, field
from typing import List
from .base import BasicEncoderStaticConfig
from ..template_modules import EncoderStaticBase
import numpy as np

class SAINT(EncoderStaticBase):
    def __init__(self, hparams_encoder_static, hparams_input_shape, static_stats_train, target_dim=None):
        super().__init__(hparams_encoder_static, hparams_input_shape, static_stats_train, target_dim)
        
        assert(np.all(np.array(hparams_encoder_static.embedding_dims)==hparams_encoder_static.embedding_dims[0]))

        output_dim = hparams_encoder_static.output_dim if target_dim is None else target_dim
        
        #https://github.com/somepago/saint/blob/main/train.py

        #if nfeat > 100:
        #    opt.embedding_size = min(8,opt.embedding_size)
        #    opt.batchsize = min(64, opt.batchsize)
        #if opt.attentiontype != 'col':
        #    opt.transformer_depth = 1
        #    opt.attention_heads = min(4,opt.attention_heads)
        #    opt.attention_dropout = 0.8
        #    opt.embedding_size = min(32,opt.embedding_size)
        #    opt.ff_dropout = 0.8
        
        self.model = SAINT(
            categories = tuple(hparams_encoder_static.vocab_sizes), 
            num_continuous = len(hparams_input_shape.static_dim),                
            dim = hparams_encoder_static.embedding_dim,                           
            dim_out = 1,                       
            depth = hparams_encoder_static.layers,                       
            heads = hparams_encoder_static.heads,                         
            attn_dropout = hparams_encoder_static.dropout_attention,             
            ff_dropout = hparams_encoder_static.dropout_ff,                  
            mlp_hidden_mults = (4, 2),       
            cont_embeddings = hparams_encoder_static.cont_embeddings,
            attentiontype = hparams_encoder_static.attention_type,
            final_mlp_style = hparams_encoder_static.final_mlp_style,
            y_dim = output_dim
            )
        
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.static_dim = output_dim
        self.output_shape.static_dim_cat = 0

        self.input_dim_cat = hparams_input_shape.static_dim_cat
    
    def forward(self, **kwargs):
        xcat = kwargs["static_cat"].long()
        xcont = kwargs["static"]
        xcat, xcont = self.layers(xcat,xcont)
        return {"static":xcont[:,0,:]}
        #TODO: check here #https://github.com/somepago/saint/blob/main/train.py
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class SAINTConfig(BasicEncoderStaticConfig):
    _target_:str = "clinical_ts.tabular.SAINT"
    embedding_dims:List[int] = field(default_factory=lambda: [32]) #list with embedding dimensions- just the first entry will be used
    
    output_dim:int = 64
    layers:int = 6
    heads:int = 8
    dropout_attention:float = 0.1
    dropout_ff:float = 0.1
    cont_embeddings:str = "MLP" #['MLP','Noemb','pos_singleMLP']
    attention_type:str = "colrow" #['col','colrow','row','justmlp','attn','attnmlp']
    final_mlp_style:str = "sep" #['common','sep']
