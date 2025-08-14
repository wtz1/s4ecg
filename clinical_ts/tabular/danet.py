__all__ = ['DANet', 'DANetConfig']

from .danet_modules.danet import DANet

import dataclasses
from dataclasses import dataclass

from .base import BasicEncoderStatic, BasicEncoderStaticConfig

class DANet(BasicEncoderStatic):
    def __init__(self, hparams_encoder_static, hparams_input_shape, static_stats_train, target_dim=None):
        super().__init__(hparams_encoder_static, hparams_input_shape, static_stats_train, target_dim)
        
        self.layers = DANet(input_dim=self.input_dim, 
                            num_classes=hparams_encoder_static.output_dim if target_dim is None else target_dim, 
                            layer_num=hparams_encoder_static.layers, 
                            base_outdim=hparams_encoder_static.base_output_dim, 
                            k=hparams_encoder_static.masks, 
                            virtual_batch_size=hparams_encoder_static.virtual_batch_size, 
                            drop_rate=hparams_encoder_static.dropout)
        
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.static_dim = int(hparams_encoder_static.output_dim if target_dim is None else target_dim)
        self.output_shape.static_dim_cat = 0
    
    def forward(self, **kwargs):
        res = self.embed(**kwargs)
        return {"static": self.layers(res)}
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class DANetConfig(BasicEncoderStaticConfig):
    _target_:str = "clinical_ts.tabular.DANet"
    output_dim:int = 64
    layers:int = 20
    masks:int = 5
    base_output_dim:int = 64
    virtual_batch_size: int = 256
    dropout:float = 0.1
    