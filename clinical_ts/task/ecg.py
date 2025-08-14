from ..template_model import SSLModel
from ..template_modules import TaskConfig
from ..data.time_series_dataset_utils import load_dataset
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .mimic_ecg_preprocessing import prepare_mimic_ecg

class ECGModel(SSLModel):
    '''class for short ECG classification tasks'''
   
    def preprocess_dataset(self,dataset_kwargs):
        df_mapped, lbl_itos, mean, std = load_dataset(Path(dataset_kwargs.path))
        # always use PTB-XL stats
        mean = np.array([-0.00184586, -0.00130277,  0.00017031, -0.00091313, -0.00148835,  -0.00174687, -0.00077071, -0.00207407,  0.00054329,  0.00155546,  -0.00114379, -0.00035649])
        std = np.array([0.16401004, 0.1647168 , 0.23374124, 0.33767231, 0.33362807,  0.30583013, 0.2731171 , 0.27554379, 0.17128962, 0.14030828,   0.14606956, 0.14656108])
        
        def multihot_encode(x, num_classes):
                res = np.zeros(num_classes,dtype=np.float32)
                for y in x:
                    res[y]=1
                return res
        
        #specific for PTB-XL
        if(self.hparams.loss.loss_type=="supervised" and dataset_kwargs.name.startswith("ptbxl")):    
            if(dataset_kwargs.name=="ptbxl_super"):
                ptb_xl_label = "label_diag_superclass"#check filtered
            elif(dataset_kwargs.name=="ptbxl_sub"):
                ptb_xl_label = "label_diag_subclass"
            elif(dataset_kwargs.name=="ptbxl_all"):
                ptb_xl_label = "label_all"
            else:
                assert(False)
                
            lbl_itos= np.array(lbl_itos[ptb_xl_label])
            df_mapped[dataset_kwargs.col_lbl]= df_mapped[ptb_xl_label+"_filtered_numeric"].apply(lambda x: multihot_encode(x,len(lbl_itos)))
        elif(self.hparams.loss.loss_type=="supervised" and dataset_kwargs.name.startswith("praxis")):
            if(dataset_kwargs.name=="praxis_abnormal"):
                lbl_itos= np.array(["normal","abnormal"])
                df_mapped[dataset_kwargs.col_lbl]= df_mapped["diag_all"].apply(lambda x: 1 if "abnorm" in x else 0)
            elif(dataset_kwargs.name=="praxis_all"): #diag_all
                lbl_itos= np.array(lbl_itos["diag_all_filtered"])
                df_mapped[dataset_kwargs.col_lbl]= df_mapped["diag_all_filtered_numeric"].apply(lambda x: multihot_encode(x,len(lbl_itos)))
            else:
                assert(False)
        elif(self.hparams.loss.loss_type=="supervised" and dataset_kwargs.name.startswith("mimic")):
            #now request the actual labels (will load records_w_diag_icd10.csv to fetch the full labels)
            df_mapped, lbl_itos = prepare_mimic_ecg(dataset_kwargs.name, Path(dataset_kwargs.path), df_mapped=df_mapped)
        elif(self.hparams.loss.loss_type=="supervised" and dataset_kwargs.name == "code"):
            df_mapped = df_mapped[df_mapped.strat_fold>=0].copy()#select on labeled subset (-1 is unlabeled)
            df_mapped[dataset_kwargs.col_lbl]= df_mapped[dataset_kwargs.col_lbl].apply(lambda x: multihot_encode(x,len(lbl_itos))) #multi-hot encode

        return df_mapped, lbl_itos, mean, std

@dataclass
class TaskConfigECG(TaskConfig):
    mainclass:str= "clinical_ts.task.ecg.ECGModel"

############################################################################################################
class ECGSeqModel(SSLModel):
    '''ECG tasks with sequence labels'''
    pass

@dataclass
class TaskConfigECGSeq(TaskConfig):
    mainclass:str= "clinical_ts.task.ecg.ECGSeqModel"


# ------------------------------------------------------------
# ★ 额外 import —— 只影响本文件，不触碰全局配置
# ------------------------------------------------------------
from ..data.time_series_dataset import tsdata_collate_fn
from torch.utils.data import DataLoader


class ECGSeqFinalTestModel(ECGSeqModel):
    """只跑最终 test_dataset_test；让 test_dataset_val 指向同一份数据，避免 AttributeError."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 只留 1 槽，父类不会再访问 idx=1
        self.test_preds = [[]]
        self.test_targs = [[]]

    # ----------------------------------------------------------
    # dataset  /  dataloader
    # ----------------------------------------------------------
    def setup(self, stage: str) -> None:
        super().setup(stage)

        # 把 val-test 映射到同一对象，省事省内存
        self.test_dataset_val = self.test_dataset_test

    def test_dataloader(self):
        """Lightning 仍然看到“一个 dataloader”，但它的下标仍是 0。"""
        return DataLoader(
            self.test_dataset_test,
            collate_fn=tsdata_collate_fn,
            batch_size=self.hparams.base.batch_size,
            num_workers=self.hparams.trainer.num_workers,
        )

    # 只处理 idx=0 的结果，避免父类循环溢出
    def on_test_epoch_end(self):
        self.on_valtest_epoch_end(
            {"preds": self.test_preds[0], "targs": self.test_targs[0]},
            test=True,
            dataloader_idx=0,
        )
        self.test_preds[0].clear()
        self.test_targs[0].clear()


@dataclass
class TaskConfigECGSeqFinalTest(TaskConfig):
    mainclass: str = "clinical_ts.task.ecg.ECGSeqFinalTestModel"
