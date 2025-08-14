__all__ = ['MambaPredictor','MambaPredictorConfig']
#pip install mamba-ssm and causal-conv1d

from .mamba_modules.mamba_model import MixerModel
from ..template_modules import PredictorBase, PredictorBaseConfig
from dataclasses import dataclass

class MambaPredictor(PredictorBase):
    def __init__(self, hparams_predictor, hparams_input_shape):
        super().__init__(hparams_predictor, hparams_input_shape)
        
        self.predictor = MixerModel(
            d_model = hparams_predictor.model_dim,
            n_layer = hparams_predictor.layers,
            vocab_size= 0,
            ssm_cfg ={"d_state": hparams_predictor.state_dim},
            continuous_input=True,
            d_input=hparams_input_shape.channels if hparams_input_shape.channels!=hparams_predictor.model_dim else None) #note: only apply linear layer before if feature dimensions do not match

    def forward(self, **kwargs):   
        return {"seq": self.predictor(kwargs["seq"])}

@dataclass
class MambaPredictorConfig(PredictorBaseConfig):
    _target_:str = "clinical_ts.ts.mamba.MambaPredictor"
    model_dim:int = 2560 
    state_dim:int = 64
    layers:int = 64
    