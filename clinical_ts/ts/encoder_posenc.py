__all__ = ['EncoderPosEnc','EncoderPosEncConfig', 'RNNEncoderPosEncConfig']

from ..template_modules import EncoderBase, EncoderBaseConfig, _string_to_class
from dataclasses import dataclass, field
import torch
from .transformer_modules.transformer import LearnedPositionalEncoding, PositionalEncoding
from .encoder import RNNEncoderConfig

class EncoderPosEnc(EncoderBase):
    '''Encoder wrapper that adds a positional encoding (to provide absolute positions for the predictor)'''
    def __init__(self, hparams_encoder, hparams_input_shape, static_stats_train):
        assert(hparams_encoder.max_len>0)#maximum input size has to be set explicitly (normally sequence length)
        assert(hparams_encoder.post_enc or hparams_input_shape.channels2==0)#no spectrogram input
        super().__init__(hparams_encoder, hparams_input_shape, static_stats_train)
        
        self.post_enc = hparams_encoder.post_enc
        self.sequence_last = hparams_input_shape.sequence_last

        self.enc = _string_to_class(hparams_encoder.enc._target_)(hparams_encoder.enc, hparams_input_shape)
        self.output_shape = self.enc.get_output_shape()
        
        if(hparams_encoder.posenc=="sine"):
            self.posenc  = PositionalEncoding(self.output_shape.channels if self.post_enc else hparams_input_shape.channels, hparams_encoder.posenc_dropout,hparams_encoder.max_len)
        elif(hparams_encoder.posenc=="learned"):
            self.posenc = LearnedPositionalEncoding(self.output_shape.channels if self.post_enc else hparams_input_shape.channels, hparams_encoder.posenc_dropout,hparams_encoder.max_len)
        else:
            assert(False)
    
    def __str__(self):
        return self.__class__.__name__+"("+self.enc.__class__.__name__+")"+"\toutput shape:"+str(self.get_output_shape())
    
    def forward(self, **kwargs):    
        if(not self.post_enc and self.sequence_last):
            kwargs["seq"] = torch.movedim(kwargs["seq"],-1,1)
        #seq has shape bs,seq,feat
        if(self.post_enc):
            seq = self.enc(**kwargs)["seq"]
        else:
            seq = kwargs["seq"]

        diff = kwargs["seq_idxs"][0,2]-kwargs["seq_idxs"][0,1] #seq_idxs has shape bs,3 where :,0 dataset idx, :,1 start idx and :,2 end idx
        idxs = torch.arange(0,diff,diff//seq.size(1),dtype=torch.int64,device=seq.device).repeat(seq.size(0),1) #bs,seq
        idxs = idxs + kwargs["seq_idxs"][:,1].unsqueeze(-1) #add start idxs
        seq = self.posenc(torch.movedim(seq,0,1),idxs) #pos emb expects seq,bs,feat
        seq = torch.movedim(seq,0,1)

        if(self.post_enc):
            return {"seq":seq}
        else:
            if(self.sequence_last):#make sure the encoder receives input in the expected format
                kwargs["seq"] = torch.movedim(seq,1,-1)
            else:
                kwargs["seq"] = seq
            return self.enc(**kwargs)
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class EncoderPosEncConfig(EncoderBaseConfig):
    _target_:str = "clinical_ts.ts.encoder_posenc.EncoderPosEnc"
    enc:EncoderBaseConfig =  field(default_factory=EncoderBaseConfig)
    posenc:str = "sine" #sine/learned
    posenc_dropout:float = 0.1
    max_len:int=0#largest idx to be encountered HAS TO BE FILLED
    post_enc:bool=True#place positional embedding post/pre encoder
    
@dataclass
class RNNEncoderPosEncConfig(EncoderPosEncConfig):
    enc:EncoderBaseConfig =  field(default_factory=RNNEncoderConfig)
