__all__ = ['Resnet1dPredictor','Resnet1dPredictorConfig','Resnet1d50PredictorConfig','Resnet1d101PredictorConfig']

from .xresnet_modules.xresnet1d import _xresnet1d
from ..template_modules import PredictorBase, PredictorBaseConfig
from typing import List
import dataclasses
from dataclasses import dataclass, field

class Resnet1dPredictor(PredictorBase):
    def __init__(self, hparams_predictor, hparams_input_shape):
        super().__init__(hparams_predictor, hparams_input_shape)
        self.predictor = _xresnet1d(
                expansion=hparams_predictor.expansion,
                layers=hparams_predictor.layers,
                input_channels=hparams_input_shape.channels,
                stem_szs=hparams_predictor.stem_szs,
                input_size=hparams_input_shape.length,
                heads=hparams_predictor.heads,
                mhsa=hparams_predictor.mhsa,
                kernel_size=hparams_predictor.kernel_size,
                kernel_size_stem=hparams_predictor.kernel_size_stem,
                widen=hparams_predictor.widen,
                model_dim=hparams_predictor.model_dim,
                num_classes=None)
        

        self.output_shape = dataclasses.replace(hparams_input_shape)
        shape = self.predictor.get_output_shape()
        self.output_shape.channels = shape[0]
        self.output_shape.length = shape[1]
        
    def forward(self, **kwargs):   
        return {"seq": self.predictor(kwargs["seq"].transpose(1,2)).transpose(1,2)}
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class Resnet1dPredictorConfig(PredictorBaseConfig):
    _target_:str = "clinical_ts.ts.xresnet.Resnet1dPredictor"
    expansion:int = 4
    layers:List[int] = field(default_factory=lambda: [3,4,6,3])
    stem_szs:List[int] = field(default_factory=lambda: [32,32,64])
    heads:int=4
    mhsa:bool=False 
    kernel_size:int=5
    kernel_size_stem:int=5
    widen:float = 1.0
    model_dim:int = 256

@dataclass
class Resnet1d50PredictorConfig(Resnet1dPredictorConfig):
    layers:List[int] = field(default_factory=lambda: [3,4,6,3])

@dataclass
class Resnet1d101PredictorConfig(Resnet1dPredictorConfig):
    layers:List[int] = field(default_factory=lambda: [3,4,23,3])


    