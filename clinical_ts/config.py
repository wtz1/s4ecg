from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

#for base configs
from .template_modules import * 

#for specific configs
from .ts.encoder import *
from .ts.encoder_epoch import *
from .ts.head import *
from .ts.base import *
from .ts.rnn import *
from .ts.xresnet import *
from .ts.multires import *
from .ts.transformer import *
from .ts.encoder_spec import *
from .ts.spacetime import *
from .ts.encoder_posenc import *
from .ts.encoder_spec_mfcc import *

##################################################################
# try to import (optional) modules with special dependencies
S4_AVAILABLE = True
try:
    from .ts.s4 import *
except:
    print("WARNING: Could not import s4 module")
    S4_AVAILABLE = False

MAMBA_AVAILABLE = True
try:
    from .ts.mamba import *
except:
    print("WARNING: Could not import mamba module")
    MAMBA_AVAILABLE = False

XLSTM_AVAILABLE = True
try:
    from .ts.xlstm import *
except:
    print("WARNING: Could not import xlstm module")
    XLSTM_AVAILABLE = False

TEXT_ENCODERS_AVAILABLE = True
try:
    from .text.base import *
except:
    print("WARNING: Could not import text encoders")
    TEXT_ENCODERS_AVAILABLE = False

EMA_AVAILABLE = True
from .ts.ema import *
try:
    from .ts.ema import *
except:
    print("WARNING: Could not import ema module")
    EMA_AVAILABLE = False

from .tabular.base import *
from .tabular.saint import *
from .tabular.danet import *
from .tabular.realmlp import *

from .head.multimodal import *

from .quantizer.multi import *

from .loss.supervised import *
from .loss.selfsupervised import *

from .metric.base import *

from .task.clip import *
from .task.ecg import *
from .task.psg import *
from .task.ppg import *
from .task.nmr import *
from .task.multimodal import *

###########################################################################################################
# https://hydra.cc/docs/tutorials/structured_config/config_groups/
@dataclass
class FullConfig:

    base: BaseConfig
    data: BaseConfigData
    loss: LossConfig
    metric: MetricConfig
    trainer: TrainerConfig
    task: TaskConfig

    ts: TimeSeriesEncoderConfig
    static: EncoderStaticBaseConfig
    head: HeadBaseConfig
    

def create_default_config():
    cs = ConfigStore.instance()
    cs.store(name="config", node=FullConfig)

    ######################################################################
    # base
    ######################################################################
    cs.store(group="base", name="base", node=BaseConfig)

    ######################################################################
    # input data
    ######################################################################
    cs.store(group="data", name="base", node=BaseConfigData)
    
    ######################################################################
    # time series encoder
    ######################################################################
    cs.store(group="ts", name="tsenc",  node=TimeSeriesEncoderConfig)
    
    #ENCODER
    cs.store(group="ts/enc", name="none", node=NoEncoderConfig)
    cs.store(group="ts/enc", name="rnn", node=RNNEncoderConfig)
    cs.store(group="ts/enc", name="tf", node=TransformerEncoderConfig)
    cs.store(group="ts/enc", name="spec2d", node=SpecConv2dEncoderConfig)#for spectrogram input
    cs.store(group="ts/enc", name="spec1d", node=SpecConv1dEncoderConfig)#for spectrogram input
    #cs.store(group="ts/enc", name="patch", node=TransformerPatchEncoderConfig) #can be emulated using the conv encoder
    #epoch encoders
    cs.store(group="ts/enc", name="pos", node=EncoderPosEncConfig)
    cs.store(group="ts/enc", name="rnnpos", node=RNNEncoderPosEncConfig)

    cs.store(group="ts/enc", name="spec", node=SpectrogramEncoderConfig)
    cs.store(group="ts/enc", name="mfcc", node=MFCCEncoderConfig)

    cs.store(group="ts/enc", name="ts", node=TimeSeriesEncoderConfig)#generic time series encoder

    cs.store(group="ts/enc", name="eernn", node=EpochEncoderRNNConfig)#CNN+LSTM epoch encoder
    cs.store(group="ts/enc", name="eetf", node=EpochEncoderTransformerConfig)#CNN+TF epoch encoder
    cs.store(group="ts/enc", name="ees4", node=EpochEncoderS4Config)#CNN+S4 epoch encoder
    cs.store(group="ts/enc", name="eens4", node=EpochEncoderNoneS4Config)#NONE+S4 epoch encoder (e.g. for use with spectrograms)
    cs.store(group="ts/enc", name="ees4cpc", node=EpochEncoderS4CPCConfig)#CNN+S4 epoch encoder with CPC loss
    
    #PREDICTOR
    cs.store(group="ts/pred", name="none", node=NoPredictorConfig)#no predictor
    cs.store(group="ts/pred", name="rnn", node=RNNPredictorConfig)#LSTM/GRU
    cs.store(group="ts/pred", name="tf", node=TransformerPredictorConfig)#transformer
    if(S4_AVAILABLE):
        cs.store(group="ts/pred", name="s4", node=S4PredictorConfig)#S4 model
    cs.store(group="ts/pred", name="resnet", node=Resnet1dPredictorConfig)#xresnet
    cs.store(group="ts/pred", name="resnet50", node=Resnet1d50PredictorConfig)#xresnet
    cs.store(group="ts/pred", name="resnet101", node=Resnet1d101PredictorConfig)#xresnet
    cs.store(group="ts/pred", name="multires", node=MRPredictorConfig)#multires conv
    cs.store(group="ts/pred", name="spacetime", node=STPredictorConfig)#spacetime
    cs.store(group="ts/pred", name="cnn", node=CNNPredictorConfig)#basic cnn
    if(MAMBA_AVAILABLE):
        cs.store(group="ts/pred", name="mamba", node=MambaPredictorConfig)#mamba
    if(XLSTM_AVAILABLE):
        cs.store(group="ts/pred", name="xlstm", node=XLSTMPredictorConfig)#xlstm
    
    #HEADS
    cs.store(group="ts/head", name="none", node=HeadBaseConfig)
    #single output token prediction heads
    cs.store(group="ts/head", name="rnn", node=RNNHeadConfig)
    cs.store(group="ts/head", name="tfg", node=TransformerHeadGlobalConfig)
    cs.store(group="ts/head", name="attn", node=AttentionPoolingHeadConfig)
    cs.store(group="ts/head", name="lqattn", node=LearnableQueryAttentionPoolingHeadConfig)
    
    #multi-token output prediction heads
    cs.store(group="ts/head", name="tfm", node=TransformerHeadMultiConfig)
    #universal heads
    cs.store(group="ts/head", name="mlp", node=MLPHeadConfig)
    cs.store(group="ts/head", name="mlpr", node=MLPRegressionHeadConfig)
    cs.store(group="ts/head", name="s4", node=S4HeadConfig)
    cs.store(group="ts/head", name="pool", node=PoolingHeadConfig)
    cs.store(group="ts/head", name="flat", node=FlattenHeadConfig)
    cs.store(group="ts/head", name="poolflat", node=PoolingFlattenHeadConfig)
    cs.store(group="ts/head", name="poolflatmlp", node=PoolingFlattenMLPHeadConfig)
    cs.store(group="ts/head", name="flatmlp", node=FlattenMLPHeadConfig)
    cs.store(group="ts/head", name="seq", node=SequentialHeadConfig)#generic sequential head (with unspecified internal heads)
    cs.store(group="ts/head", name="poolconcat", node=PoolingConcatFusionHeadConfig)

    #SSL HEADS
    cs.store(group="ts/head_ssl", name="none", node=HeadBaseConfig)
    #multi prediction heads
    cs.store(group="ts/head_ssl", name="tfm", node=TransformerHeadMultiConfig)
    #universal heads
    cs.store(group="ts/head_ssl", name="mlp", node=MLPHeadConfig)
    cs.store(group="ts/head_ssl", name="s4", node=S4HeadConfig)
    cs.store(group="ts/head_ssl", name="pool", node=PoolingHeadConfig)

    #QUANTIZER
    cs.store(group="ts/quant", name="none", node=QuantizerBaseConfig)
    cs.store(group="ts/quant", name="multi", node=MultiCodebookQuantizerConfig)

    #MASK
    cs.store(group="ts/mask", name="none", node=MaskingBaseConfig)
    cs.store(group="ts/mask", name="mask", node=MaskingConfig)

    #LOSS
    cs.store(group="ts/loss", name="none", node=SSLLossConfig)
    cs.store(group="ts/loss", name="maskrec", node=MaskedReconstructionLossConfig)
    cs.store(group="ts/loss", name="maskpredhubert", node=MaskedPredictionLossHuBERTConfig)
    cs.store(group="ts/loss", name="maskpredcapi", node=MaskedPredictionLossCAPIConfig)
    cs.store(group="ts/loss", name="cpc", node=CPCLossConfig)
    cs.store(group="ts/loss", name="clip", node=CLIPLossConfig)
    cs.store(group="ts/loss", name="infonce", node=InfoNCELossConfig)

    #PRE
    cs.store(group="ts/pre", name="none", node=PrePostBaseConfig)  
    cs.store(group="ts/pre", name="ee", node=EpochEncoderPreConfig)
    
    #POST
    cs.store(group="ts/pre", name="none", node=PrePostBaseConfig)  
    cs.store(group="ts/post", name="ee", node=EpochEncoderPostConfig)

    #EMA
    cs.store(group="ts/ema", name="none", node=EMATimeSeriesEncoderBaseConfig)
    if(EMA_AVAILABLE):
        cs.store(group="ts/ema", name="ema", node=EMATimeSeriesEncoderConfig)
    
    ######################################################################
    # static encoder
    ######################################################################
    for g in ["static", "ts/static"]:
        cs.store(group=g, name="none", node=EncoderStaticBaseConfig)
        cs.store(group=g, name="mlp", node=BasicEncoderStaticMLPConfig)
        cs.store(group=g, name="saint", node=SAINTConfig)
        cs.store(group=g, name="danet", node=DANetConfig)
        cs.store(group=g, name="realmlp", node=RealMLPConfig)
        
        if(TEXT_ENCODERS_AVAILABLE):
            cs.store(group=g, name="text", node=TransformerTextEncoderConfig)
            cs.store(group=g, name="pretext", node=PretrainedTextEncoderConfig)

    ######################################################################
    # optional multimodal head
    ######################################################################
    cs.store(group="head", name="none", node=HeadBaseConfig)
    cs.store(group="head", name="rnn", node=RNNHeadConfig)
    cs.store(group="head", name="concat", node=ConcatFusionHeadConfig)
    cs.store(group="head", name="tensor", node=TensorFusionHeadConfig)
    cs.store(group="head", name="attn", node=AttentionFusionHeadConfig)
    cs.store(group="head", name="poolconcat", node=PoolingConcatFusionHeadConfig)

    ######################################################################
    # loss function
    ######################################################################
    #no global loss
    cs.store(group="loss", name="none", node=LossConfig)
    #supervised losses
    cs.store(group="loss", name="ce", node=CELossConfig)
    cs.store(group="loss", name="cef", node=CEFLossConfig)
    cs.store(group="loss", name="bce", node=BCELossConfig)
    cs.store(group="loss", name="bcef", node=BCEFLossConfig)
    cs.store(group="loss", name="qreg", node=QuantileRegressionLossConfig)
    cs.store(group="loss", name="mse", node=MSELossConfig)
    
    #global ssl losses
    cs.store(group="loss", name="clip", node=CLIPLossConfig)
    cs.store(group="loss", name="infonce", node=InfoNCELossConfig)   
    
    ######################################################################
    # metrics
    ######################################################################
    cs.store(group="metric", name="none", node=MetricConfig)
    cs.store(group="metric", name="auroc", node=MetricAUROCConfig)
    cs.store(group="metric", name="aurocagg", node=MetricAUROCAggConfig)
    cs.store(group="metric", name="aupr", node=MetricAUPRConfig)
    cs.store(group="metric", name="aupragg", node=MetricAUPRAggConfig)
    cs.store(group="metric", name="mae", node=MetricMAEConfig)
    cs.store(group="metric", name="maeagg", node=MetricMAEAggConfig)
    cs.store(group="metric", name="fbeta", node=MetricFbetaConfig)
    cs.store(group="metric", name="fbetaagg", node=MetricFbetaAggConfig)
    cs.store(group="metric", name="f1", node=MetricF1Config)
    cs.store(group="metric", name="f1agg", node=MetricF1AggConfig)
    cs.store(group="metric", name="acc", node=MetricAccuracyConfig)
    cs.store(group="metric", name="accagg", node=MetricAccuracyAggConfig)
    cs.store(group="metric", name="sensspec", node=MetricSensitivitySpecificityConfig)
    cs.store(group="metric", name="sensspecagg", node=MetricSensitivitySpecificityAggConfig)

    ######################################################################
    # trainer
    ######################################################################
    cs.store(group="trainer", name="trainer", node=TrainerConfig)
    
    ######################################################################
    # task
    ######################################################################
    cs.store(group="task", name="none", node=TaskConfig)
    cs.store(group="task", name="psg", node=TaskConfigPSG)
    cs.store(group="task", name="ppg", node=TaskConfigPPG)
    cs.store(group="task", name="ecg", node=TaskConfigECG)
    cs.store(group="task", name="ecgseq", node=TaskConfigECGSeq)
    cs.store(group="task", name="clip", node=TaskConfigCLIP)
    cs.store(group="task", name="nmr", node=TaskConfigNMR)
    cs.store(group="task", name="multi", node=TaskConfigMultimodal)
    cs.store(group="task", name="ecgtest", node=TaskConfigECGSeqFinalTest)
    
    return cs
