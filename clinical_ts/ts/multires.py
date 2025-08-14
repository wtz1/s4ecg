__all__ = ['MRPredictor','MRPredictorConfig']

from .multires_modules.multires import MultiresModel
from ..template_modules import PredictorBase, PredictorBaseConfig
from dataclasses import dataclass

class MRPredictor(PredictorBase):
    def __init__(self, hparams, hparams_input_shape):
        super().__init__(hparams, hparams_input_shape)
        self.predictor = MultiresModel(
            d_input = hparams_input_shape.channels if hparams_input_shape.channels!=hparams.predictor.model_dim else None,#modified
            d_output=None,
            d_model=hparams.predictor.model_dim,
            n_layers=hparams.predictor.layers,
            dropout=hparams.predictor.dropout,
            layer_norm=not(hparams.predictor.batchnorm),
            layer_type="multires",
            l_max=hparams_input_shape.length,
            hinit=None,
            depth=None,
            tree_select="fading",
            d_mem=None,
            kernel_size=2,
            indep_res_init=False,
            transposed_input=False, # behaves like 1d CNN if True else like a RNN with batch_first=True
            pooling=False)
        
    def forward(self, **kwargs):
        return {"seq":  self.predictor(kwargs["seq"])}

@dataclass
class MRPredictorConfig(PredictorBaseConfig):
    _target_:str = "clinical_ts.ts.multires.MRPredictor"
    model_dim:int = 128 
    causal: bool = True #use bidirectional predictor
    layers:int = 4
    dropout:float=0.1
    batchnorm:bool=False
