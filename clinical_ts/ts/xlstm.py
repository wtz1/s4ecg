__all__ = ['XLSTMPredictor','XLSTMPredictorConfig']

from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig

from ..template_modules import PredictorBase, PredictorBaseConfig
from dataclasses import dataclass
from torch import nn

class XLSTMPredictor(PredictorBase):
    def __init__(self, hparams_predictor, hparams_input_shape):
        super().__init__(hparams_predictor, hparams_input_shape)

        self.input_mapping = nn.Linear(hparams_input_shape.channels,hparams_predictor.model_dim) if hparams_input_shape.channels!= hparams_predictor.model_dim else None

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                      conv1d_kernel_size=hparams_predictor.conv1d_kernel_size, 
                      qkv_proj_blocksize=4, 
                      num_heads=hparams_predictor.num_heads)),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=hparams_predictor.num_heads,
                    conv1d_kernel_size=hparams_predictor.conv1d_kernel_size,
                    bias_init="powerlaw_blockdependent",),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),),
            context_length=hparams_input_shape.length,
            num_blocks=hparams_predictor.num_blocks,
            embedding_dim=self.model_dim,
            slstm_at=[1],)

        self.predictor = xLSTMBlockStack(cfg)

    def forward(self, **kwargs):
        seq = kwargs["seq"]
        if(self.input_mapping is not None):
            seq = self.input_mapping(seq)
        #xlstm has input/output shape bs,seq,emb
        return {"seq": self.predictor(kwargs["seq"])}

@dataclass
class XLSTMPredictorConfig(PredictorBaseConfig):
    _target_:str = "clinical_ts.ts.xlstm.XLSTMPredictor"
    model_dim:int = 128
    conv1d_kernel_size:int = 4
    num_heads:int = 4
    num_blocks:int = 7