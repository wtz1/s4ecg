__all__ = ['STPredictor','STPredictorConfig']

from ..template_modules import PredictorBase, PredictorBaseConfig
from typing import Any
from dataclasses import dataclass

from .spacetime_modules.network import SpaceTime

class STPredictor(PredictorBase):
    def __init__(self, hparams, hparams_input_shape):
        super().__init__(hparams, hparams_input_shape)

        input_dim = hparams_input_shape.channels
        output_dim = None
        model_dim = hparams.predictor.model_dim
        kernel_dim = hparams.predictor.kernel_dim
        layers_encoder = hparams.predictor.layers_encoder
        layers_decoder = hparams.predictor.layers_decoder
        norm_order = hparams.predictor.norm_order
        dropout  = hparams.predictor.dropout
        final_activation = hparams.predictor.final_activation

        lag = hparams.predictor.lag
        horizon = hparams.predictor.horizon
        inference_only = hparams.predictor.inference_only

        #embedding (only use one if the dimensionality does not match)
        if(input_dim!=model_dim):
            embedding_config = {'method': 'linear', 'kwargs': {'input_dim': input_dim, 'embedding_dim': model_dim}}
            #embedding_config = {'method': 'repeat', 'kwargs': {'input_dim': input_dim, 'embedding_dim': model_dim, 'n_heads': 4, 'n_kernels': 32}}
        else:
            embedding_config = {'method': 'identity', 'kwargs': {'input_dim': input_dim, 'embedding_dim': model_dim}}

        #encoder
        encoder_block_start = {'input_dim': model_dim, 'pre_config': {'method': 'residual', 'kwargs': {'max_diff_order': 4, 'min_avg_window': 4, 'max_avg_window': 64, 'model_dim': model_dim, 'n_kernels': 8, 'kernel_dim': 2, 'kernel_repeat': model_dim//8, 'n_heads': 1, 'head_dim': 1, 'kernel_weights': None, 'kernel_init': None, 'kernel_train': False, 'skip_connection': False, 'seed': 0}}, 'ssm_config': {'method': 'companion', 'kwargs': {'model_dim': model_dim, 'n_kernels': 16, 'kernel_dim': kernel_dim, 'kernel_repeat': 1, 'n_heads': None, 'head_dim': 1, 'kernel_weights': None, 'kernel_init': 'normal', 'kernel_train': True, 'skip_connection': True, 'norm_order': norm_order}}, 'mlp_config': {'method': 'mlp', 'kwargs': {'input_dim': model_dim, 'output_dim': model_dim, 'activation': 'gelu', 'dropout': dropout, 'layernorm': False, 'n_layers': 1, 'n_activations': 1, 'pre_activation': True, 'input_shape': 'bld', 'skip_connection': True, 'average_pool': None}}, 'skip_connection': True, 'skip_preprocess': False}
        encoder_block_standard = {'input_dim': model_dim, 'pre_config': {'method': 'identity', 'kwargs': None}, 'ssm_config': {'method': 'companion', 'kwargs': {'model_dim': model_dim, 'n_kernels': model_dim, 'kernel_dim': kernel_dim, 'kernel_repeat': 1, 'n_heads': 1, 'head_dim': 1, 'kernel_weights': None, 'kernel_init': 'normal', 'kernel_train': True, 'skip_connection': True, 'norm_order': norm_order}}, 'mlp_config': {'method': 'mlp', 'kwargs': {'input_dim': model_dim, 'output_dim': model_dim, 'activation': 'gelu', 'dropout': dropout, 'layernorm': False, 'n_layers': 1, 'n_activations': 1, 'pre_activation': True, 'input_shape': 'bld', 'skip_connection': True, 'average_pool': None}}, 'skip_connection': True, 'skip_preprocess': True}
        encoder_config= {'blocks': [encoder_block_start]+ [encoder_block_standard]*(layers_encoder-1)}
        if(layers_decoder==0 and output_dim is None and not final_activation):
            encoder_config["blocks"][-1]["mlp_config"]= {'method': 'identity', 'kwargs': None} #activation is taken care of in output layer

        #decoder
        decoder_block_closed = {'input_dim': model_dim, 'pre_config': {'method': 'identity', 'kwargs': None}, 'ssm_config': {'method': 'closed_loop_companion', 'kwargs': {'lag':lag, 'horizon':horizon, 'model_dim': model_dim, 'n_kernels': model_dim, 'kernel_dim': kernel_dim, 'kernel_repeat': 1, 'n_heads': 1, 'head_dim': 1, 'kernel_weights': None, 'kernel_init': 'normal', 'kernel_train': True, 'skip_connection': False, 'norm_order': norm_order, 'use_initial':False}}, 'mlp_config': {'method': 'mlp', 'kwargs': {'input_dim': model_dim, 'output_dim': model_dim, 'activation': 'gelu', 'dropout': dropout, 'layernorm': False, 'n_layers': 1, 'n_activations': 1, 'pre_activation': True, 'input_shape': 'bld', 'skip_connection': True, 'average_pool': None}}, 'skip_connection': False, 'skip_preprocess': False}
        decoder_config={'blocks': [] if layers_decoder==0 else [decoder_block_closed]+[encoder_block_standard]*(layers_decoder-1)}
        if(layers_decoder>0 and output_dim is None and not final_activation):
            decoder_config["blocks"][-1]["mlp_config"]= {'method': 'identity', 'kwargs': None} #activation is taken care of in output layer

        #output
        if(output_dim is not None and final_activation):
            output_config={'input_dim': model_dim, 'output_dim': output_dim, 'method': 'mlp', 'kwargs': {'input_dim': model_dim, 'output_dim': output_dim, 'activation': 'gelu', 'dropout': dropout, 'layernorm': False, 'n_layers': 1, 'n_activations': 1, 'pre_activation': True, 'input_shape': 'bld', 'skip_connection': False, 'average_pool': None}}
        else:
            output_config={'method': 'identity', 'kwargs': None}

        # put everything together
        spacetime_config={'embedding_config': embedding_config, 'encoder_config': encoder_config, 'decoder_config': decoder_config, 'output_config': output_config, "inference_only": inference_only, "lag": lag, "horizon": horizon}


        self.predictor = SpaceTime(**spacetime_config)
    
    def forward(self, **kwargs): 
        #original model (matching the expected shape of the base class)
        #input shape B L E
        #output shape B L E

        ys, zs = self.predictor(kwargs["seq"])
        return {"seq": ys[0]}

@dataclass
class STPredictorConfig(PredictorBaseConfig):
    _target_:str = "clinical_ts.ts.spacetime.STPredictor"

    model_dim:int = 128
    kernel_dim:int = 64
    layers_encoder:int = 3
    layers_decoder:int = 0
    norm_order:int=1
    dropout:float=0.25
    final_activation:bool=True

    lag:int=900
    horizon:int=100
    inference_only:bool = False  
