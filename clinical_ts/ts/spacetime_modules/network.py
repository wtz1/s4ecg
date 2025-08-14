"""
SpaceTime Network
"""
import torch.nn as nn

from .embedding import init_embedding
from .block import Encoder, Decoder
from .mlp import init_mlp


class SpaceTime(nn.Module):
    def __init__(self,
                 embedding_config: dict,
                 encoder_config: dict,
                 decoder_config: dict,
                 output_config: dict,
                 inference_only: bool=False,
                 lag: int=1,
                 horizon: int=1):
        super().__init__()
        
        self.embedding_config  = embedding_config
        self.encoder_config    = encoder_config
        self.decoder_config    = decoder_config
        self.output_config     = output_config
        
        self.inference_only = inference_only
        self.lag     = lag
        self.horizon = horizon
        
        self.init_weights(embedding_config, encoder_config,
                          decoder_config, output_config)
        
    # -----------------
    # Initialize things
    # -----------------
    def init_weights(self, 
                     embedding_config: dict, 
                     encoder_config: dict, 
                     decoder_config: dict, 
                     output_config: dict):
        self.embedding  = self.init_embedding(embedding_config)
        self.encoder    = self.init_encoder(encoder_config)
        #MODIFIED
        self.decoder    = self.init_decoder(decoder_config) if len(decoder_config["blocks"])>0 else None
        self.output     = self.init_output(output_config)
        
    def init_embedding(self, config):
        return init_embedding(config)
    
    def init_encoder(self, config):
        self.encoder = Encoder(config)
        # Allow access to first encoder SSM kernel_dim
        self.kernel_dim = self.encoder.blocks[0].ssm.kernel_dim
        return self.encoder
    
    def init_decoder(self, config):
        self.decoder = Decoder(config)
        self.decoder.closed_block.ssm.lag = self.lag #MODIFIED
        self.decoder.closed_block.ssm.horizon = self.horizon #MODIFIED
        return self.decoder
    
    def init_output(self, config):
        return init_mlp(config)
    
    # -------------
    # Toggle things
    # -------------
    def set_inference_only(self, mode=False):
        self.inference_only = mode
        if(self.decoder is not None):#MODIFIED
            self.decoder.closed_block.ssm.inference_only = mode#MODIFIED
        
    def set_closed_loop(self, mode=True):
        if(self.decoder is not None):#MODIFIED
            self.decoder.closed_block.ssm.closed_loop = mode#MODIFIED
        
    def set_train(self):
        self.train()
        
    def set_eval(self):
        self.eval()
        self.set_inference_only(mode=True)
        
    def set_lag(self, lag: int):
        if(self.decoder is not None):#MODIFIED
            self.decoder.closed_block.ssm.lag = lag#MODIFIED
        
    def set_horizon(self, horizon: int):
        if(self.decoder is not None):#MODIFIED
            self.decoder.closed_block.ssm.horizon = horizon#MODIFIED
        
    # ------------
    # Forward pass
    # ------------
    def forward(self, u):
        self.set_closed_loop(True)
        # Assume u.shape is (batch x len x dim), 
        # where len = lag + horizon
        z = self.embedding(u)
        z = self.encoder(z)
        y_c, _ = self.decoder(z)  if self.decoder is not None else z, None #MODIFIED
        y_c = self.output(y_c)  # y_c is closed-loop output

        if not self.inference_only and self.decoder is not None:  
            # Also compute outputs via open-loop
            self.set_closed_loop(False)
            y_o, z_u = self.decoder(z)
            y_o = self.output(y_o)    # y_o is "open-loop" output
            # Prediction and "ground-truth" for next-time-step 
            # layer input (i.e., last-layer output)
            z_u_pred, z_u_true = z_u  
        else:
            y_o = None
            z_u_pred, z_u_true = None, None
        # Return (model outputs), (model last-layer next-step inputs)
        return (y_c, y_o), (z_u_pred, z_u_true)
