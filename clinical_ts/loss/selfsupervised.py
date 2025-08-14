__all__ = ['MaskedLossConfig', 'MaskedPredictionLossHuBERT', 'MaskedPredictionLossHuBERTConfig','MaskedPredictionLossCAPI','MaskedPredictionLossCAPIConfig', 'MaskedReconstructionLoss', 'MaskedReconstructionLossConfig', 'MaskingModule', 'MaskingConfig', 'CPCLoss', 'CPCLossConfig', 'CLIPLoss', 'CLIPLossConfig', 'InfoNCELoss', 'InfoNCELossConfig']

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from typing import List
from dataclasses import dataclass, field
from ..template_modules import SSLLossConfig, MaskingBaseConfig


#base config class for masking
@dataclass
class MaskedLossConfig(SSLLossConfig):
    _target_:str = ""
    loss_type:str = "masked_"

class MaskedPredictionLossCAPI(nn.Module):
    '''masked prediction loss a la CAPI'''
    def __init__(self, hparams_loss):
        super().__init__()
        self.alpha = hparams_loss.alpha
        self.t = hparams_loss.temperature
        
    def forward(self,input_predicted,ema_soft_cluster_assignments,mask_ids,**kwargs):
        results_dict = {}
        
        if(len(mask_ids)==0):#no masked tokens in entire batch
            if(self.alpha ==1.):
                for i in range(len(ema_soft_cluster_assignments)):
                    results_dict.update({"loss_masked"+str(i): 0.})
            else:
                for i in range(len(ema_soft_cluster_assignments)):
                    results_dict.update({"loss_masked"+str(i): 0., "loss_nonmasked"+str(i): 0.})

        for i,(ip,ca) in enumerate(zip(input_predicted,ema_soft_cluster_assignments)):
            # ip: bs, seq, num_clusters
            # ca: bs, seq, num_clusters
            if(self.alpha==1):
                preds = ip[mask_ids[:,0],mask_ids[:,1]]
                targs = ca[mask_ids[:,0],mask_ids[:,1]]
                loss_masked = -torch.sum(targs.float() * F.log_softmax(preds.float() / self.t, dim=-1), dim=-1)
                results_dict.update({"loss_masked"+str(i): loss_masked.mean()})
            else:
                loss_noagg = -torch.sum(ca.float() * F.log_softmax(ip.float() / self.t, dim=-1), dim=-1)
                loss_masked = torch.sum(loss_noagg[mask_ids[:,0],mask_ids[:,1]])
                loss_nonmasked = torch.sum(loss_noagg)-loss_masked

                loss_masked =  self.alpha* loss_masked/len(mask_ids)
                loss_nonmasked = (1.-self.alpha)* loss_nonmasked/(ip.shape[0]*ip.shape[1]-len(mask_ids))#todo check
                
                results_dict.update({"loss_masked"+str(i): loss_masked, "loss_nonmasked"+str(i): loss_nonmasked})

        return results_dict

class MaskedPredictionLossHuBERT(nn.Module):
    '''masked prediction loss a la HuBERT'''
    def __init__(self, hparams_loss):
        super().__init__()
        self.alpha = hparams_loss.alpha
        self.t = hparams_loss.temperature
        
    def forward(self,input_predicted,ema_cluster_labels,ema_cluster_centroids,mask_ids,**kwargs):
        results_dict = {}
        
        if(len(mask_ids)==0):#no masked tokens in entire batch
            if(self.alpha ==1.):
                for i in range(len(ema_cluster_centroids)):
                    results_dict.update({"loss_masked"+str(i): 0., "metric_acc_masked"+str(i): 0.})
            else:
                for i in range(len(ema_cluster_centroids)):
                    results_dict.update({"loss_masked"+str(i): 0., "metric_acc_masked"+str(i):0, "loss_nonmasked"+str(i): 0., "metric_acc_all"+str(i):0.})

        assert(isinstance(input_predicted,list) and len(input_predicted)==len(ema_cluster_centroids))#
        
        for i,(ip,cl,cc) in enumerate(zip(input_predicted,ema_cluster_labels, ema_cluster_centroids)):
            # ip: bs, seq, feat
            # cl: bs, seq (integer from 1... num_clusters)
            # cc: num_clusters, feat
            if(self.alpha==1):
                preds = ip[mask_ids[:,0],mask_ids[:,1]]
                preds = F.normalize(preds,p=2,dim=-1)
                cc = F.normalize(cc,p=2,dim=-1)
                sims = preds@cc.transpose(1,0)/self.t
                targs = cl[mask_ids[:,0],mask_ids[:,1]]
                loss_masked = F.cross_entropy(sims, targs)
                sims_argmax = torch.argmax(sims,dim=-1)
                acc_masked = torch.mean((sims_argmax == targs).float())
                results_dict.update({"loss_masked"+str(i): loss_masked, "metric_acc_masked"+str(i): acc_masked})
            else:
                ip = F.normalize(ip,p=2,dim=-1)
                cc = F.normalize(cc,p=2,dim=-1)
                sims = (ip@cc.transpose(1,0)).transpose(1,2)/self.t # Pytorch expects class as second axis
                loss_noagg = F.cross_entropy(sims,cl,reduction='none')
                loss_masked = torch.sum(loss_noagg[mask_ids[:,0],mask_ids[:,1]])
                loss_nonmasked = torch.sum(loss_noagg)-loss_masked

                loss_masked =  self.alpha* loss_masked/len(mask_ids)
                loss_nonmasked = (1.-self.alpha)* loss_nonmasked/(sims.shape[0]*sims.shape[1]-len(mask_ids))
                
                sims_argmax = torch.argmax(sims,dim=1)
                acc_all = torch.mean((sims_argmax == cl).float())
                acc_masked = torch.mean((sims_argmax[mask_ids[:,0],mask_ids[:,1]] == cl[mask_ids[:,0],mask_ids[:,1]]).float())
                results_dict.update({"loss_masked"+str(i): loss_masked, "metric_acc_masked"+str(i):acc_masked, "loss_nonmasked"+str(i): loss_nonmasked, "metric_acc_all"+str(i):acc_all})
            
        return results_dict

@dataclass
class MaskedPredictionLossConfig(MaskedLossConfig):
    _target_:str = "clinical_ts.loss.selfsupervised.MaskedPredictionLoss"
    loss_type:str = "masked_pred"
    kmeans_ks:List[int]=field(default_factory=lambda: [50,100])
    temperature:float = 0.1
    alpha:float = 1. #1.: only loss on the masked tokens, 0. only loss on the unmasked tokens

@dataclass
class MaskedPredictionLossCAPIConfig(MaskedPredictionLossConfig):
    _target_:str = "clinical_ts.loss.selfsupervised.MaskedPredictionLossCAPI"
    loss_type:str = "masked_pred_capi"

@dataclass
class MaskedPredictionLossHuBERTConfig(MaskedPredictionLossConfig):
    _target_:str = "clinical_ts.loss.selfsupervised.MaskedPredictionLossHuBERT"
    loss_type:str = "masked_pred_hubert"

class MaskedReconstructionLoss(nn.Module):
    '''contrastive masked reconstruction loss a la wav2vec2'''
    def __init__(self, hparams_loss):
        super().__init__()
        self.n_false_negatives = hparams_loss.n_false_negatives
        self.negatives_from_same_seq_only = hparams_loss.negatives_from_same_seq_only
        self.t = hparams_loss.temperature
        self.normalize = hparams_loss.normalize

    def forward(self,input_predicted,input_encoded,mask_ids,**kwargs):
        if(len(mask_ids)==0):#no masked tokens in entire batch
            return {"loss": 0, "metric_acc": 0} 
        #both input_predicted and input_encoded have shape bs,seq,features
        positives = input_encoded[mask_ids[:,0],mask_ids[:,1]].unsqueeze(dim=1) # elements==num_masked_tokens,1,features
        negatives = []
        for i,pid in enumerate(mask_ids):#mask_id has shape elements,2 where [:,0] gives the bs index and [:,1] the seq index
            idxs_candidates = torch.where(mask_ids[:,0]==pid[0])[0] if self.negatives_from_same_seq_only else torch.arange(0,len(mask_ids)-1,device=mask_ids.device)#select ids (to index into mask_ids) from same sequence
            idxs_distractors = torch.randint(0,len(idxs_candidates)-1,(self.n_false_negatives,),device=input_predicted.device)
            #make sure we don't pick the positive
            idx_pos = torch.where(mask_ids[idxs_candidates,1]==pid[1])[0] if self.negatives_from_same_seq_only else i
            idxs_seq2 = idxs_distractors * (idxs_distractors<idx_pos).long() +(idxs_distractors+1)*(idxs_distractors>=idx_pos).long()#false_neg
            distractors_selected = idxs_candidates[idxs_seq2]
            negatives.append(input_encoded[mask_ids[distractors_selected,0],mask_ids[distractors_selected,1]]) #false negatives, features
        negatives = torch.stack(negatives,dim=0)
        candidates = torch.cat([positives,negatives],dim=1)#elements, false negatives+1, features
        preds = input_predicted[mask_ids[:,0],mask_ids[:,1]]#elements, features
        if(self.normalize):
            candidates = F.normalize(candidates, p=2.0, dim = -1)
            preds = F.normalize(preds, p=2.0, dim = -1)
        sim = torch.sum(preds.unsqueeze(1)*candidates,dim=-1)/self.t#elements, false negatives+1
        targs = torch.zeros(preds.size(0), dtype=torch.int64, device=input_predicted.device)
        #if(eval_acc):
        sim_argmax = torch.argmax(sim,dim=-1)
        tp_cnt = torch.sum(sim_argmax == targs)
                
        loss = F.cross_entropy(sim,targs)
        return {"loss": loss, "metric_acc": tp_cnt.float()/len(preds)}

@dataclass
class MaskedReconstructionLossConfig(MaskedLossConfig):
    _target_:str = "clinical_ts.loss.selfsupervised.MaskedReconstructionLoss"
    loss_type:str = "masked_rec"
    n_false_negatives:int = 100 #number of distractors in the contrastive loss (wav2vec default: 100)
    negatives_from_same_seq_only:bool = True # only draw false negatives from same sequence (as opposed to drawing from everywhere)
    temperature:float = 0.1 # temperature in the softmax (wav2vec default: 0.1)
    normalize:bool = True # normalize vectors before calculating similarities (wav2vec default True)

class MaskingModule(nn.Module):
    def __init__(self, hparams_masking, hparams_input_shape):
        super().__init__()
        self.mask_probability = hparams_masking.mask_probability/hparams_masking.mask_span
        self.mask_span = hparams_masking.mask_span
        
        self.mask_item = nn.Parameter(torch.randn([hparams_input_shape.channels]).unsqueeze(dim=0))
        self.output_shape = hparams_input_shape
        
    def forward(self, seq, **kwargs):
        #bs,seq,features
        mask = self.calculate_mask(seq)
        mask_ids = torch.nonzero(mask)
        input_encoded_masked = seq.clone()
        input_encoded_masked[torch.where(mask)] = self.mask_item.repeat([len(mask_ids),1]).to(seq.dtype)
        return {"seq":input_encoded_masked, "mask_ids": mask_ids}#all mask_ids have shape bs,ts
    
    def calculate_mask(self, seq):
        bd=torch.distributions.bernoulli.Bernoulli(self.mask_probability)
        mask_sparse=bd.sample([seq.shape[0],seq.shape[1]]).long().to(seq.device)
        mask_full = mask_sparse.clone()
        midpoints= torch.nonzero(mask_sparse==1).cpu().numpy()#midpoints
        steps_after = (self.mask_span-1)//2 if (self.mask_span-1)%2==0 else self.mask_span//2
        steps_before = self.mask_span-steps_after-1
        
        for x in midpoints:
            mask_full[x[0],max(x[1]-steps_before,0):min(x[1]+steps_after+1,mask_sparse.shape[1]-1)]=1
        return mask_full
        #return mask_full, torch.nonzero(mask_sparse==1), torch.nonzero((mask_full-mask_sparse)==1) #full mask, midpoint ids, other masked ids

    def get_output_shape(self):
        return self.output_shape
    
    def __str__(self):
        return self.__class__.__name__+"\toutput shape:"+str(self.get_output_shape())

@dataclass
class MaskingConfig(MaskingBaseConfig):
    _target_:str = "clinical_ts.loss.selfsupervised.MaskingModule"
    mask_probability:float = 0.065 #probability that a certain position in the sequence gets drawn as midpoint
    mask_span: int = 10 # draw mask_span surrounding positions around midpoints

class CPCLoss(nn.Module):
    def __init__(self, hparams_loss):
        super().__init__()
        self.steps_predicted = hparams_loss.steps_predicted
        self.n_false_negatives = hparams_loss.n_false_negatives
        self.negatives_from_same_seq_only = hparams_loss.negatives_from_same_seq_only
        self.negatives_selection_interval = hparams_loss.negatives_selection_interval
        assert(self.negatives_from_same_seq_only is False or self.negatives_selection_interval==0 or self.negatives_selection_interval*2>= self.n_false_negatives)#make sure to have enough negatives available
        self.t = hparams_loss.temperature
        self.normalize = hparams_loss.normalize
        
    def forward(self,input_predicted,input_encoded, **kwargs):
        #both: bs,seq,features
        input_encoded_flat = input_encoded.reshape(-1,input_encoded.size(2)) #for negatives below: -1, features
        
        bs = input_encoded.size()[0]
        seq = input_encoded.size()[1]
        
        tp_cnt = torch.tensor(0,dtype=torch.int64, device=input_predicted.device)
        loss = torch.tensor(0,dtype=torch.float32, device=input_predicted.device)

        for i in range(seq-self.steps_predicted):
            positives = input_encoded[:,i+self.steps_predicted].unsqueeze(1) #bs,1,encoder_output_dim
            
            start = max(0,i-self.negatives_selection_interval) if self.negatives_selection_interval>0 else 0
            end = min(i+self.negatives_selection_interval+1,seq if self.negatives_selection_interval<self.steps_predicted else seq-1) if self.negatives_selection_interval>0 else seq-1

            idxs_seq = torch.randint(start,end,(bs*self.n_false_negatives,), device=input_predicted.device)
            #make sure we don't pick the positive
            idxs_seq2 = idxs_seq * (idxs_seq<(i+self.steps_predicted)).long() +(idxs_seq+1)*(idxs_seq>=(i+self.steps_predicted)).long()#bs*false_neg
            if(self.negatives_from_same_seq_only):
                idxs_batch = torch.arange(0,bs, device=input_predicted.device).repeat_interleave(self.n_false_negatives)
            else:
                idxs_batch = torch.randint(0,bs,(bs*self.n_false_negatives,), device=input_predicted.device)
            idxs2_flat = idxs_batch*seq+idxs_seq2

            #old
            #if(self.negatives_from_same_seq_only):
            #    idxs = torch.randint(0,(seq-1),(bs*self.n_false_negatives,)).to(input_predicted.device)
            #else:#negative from everywhere
            #    idxs = torch.randint(0,bs*(seq-1),(bs*self.n_false_negatives,)).to(input_predicted.device)
            #idxs_seq = torch.remainder(idxs,seq-1) #bs*false_neg
            #idxs_seq2 = idxs_seq * (idxs_seq<(i+self.steps_predicted)).long() +(idxs_seq+1)*(idxs_seq>=(i+self.steps_predicted)).long()#bs*false_neg
            #if(self.negatives_from_same_seq_only):
            #    idxs_batch = torch.arange(0,bs).repeat_interleave(self.n_false_negatives).to(input_predicted.device)
            #else:
            #    idxs_batch = idxs//(seq-1)
            #idxs2_flat = idxs_batch*seq+idxs_seq2 #for negatives from everywhere: this skips step i+steps_predicted from the other sequences as well for simplicity
            
            negatives = input_encoded_flat[idxs2_flat].view(bs,self.n_false_negatives,-1) #bs*false_neg, encoder_output_dim
            candidates = torch.cat([positives,negatives],dim=1)#bs,false_neg+1,encoder_output_dim
            preds = input_predicted[:,i]
            if(self.normalize):
                candidates = F.normalize(candidates, p=2.0, dim = -1)
                preds = F.normalize(preds, p=2.0, dim = -1)

            sim=torch.sum(preds.unsqueeze(1)*candidates,dim=-1)/self.t #bs,(false_neg+1)
            targs = torch.zeros(bs, dtype=torch.int64, device=input_predicted.device)
            
            #if(eval_acc):
            sim_argmax = torch.argmax(sim,dim=-1)
            tp_cnt += torch.sum(sim_argmax == targs)
                
            loss += F.cross_entropy(sim,targs)
        return {"loss":loss, "metric_acc":tp_cnt.float()/bs/(input_encoded.size()[1]-self.steps_predicted)}


@dataclass
class CPCLossConfig(SSLLossConfig):
    _target_:str = "clinical_ts.loss.selfsupervised.CPCLoss"
    loss_type:str = "cpc"
    steps_predicted: int = 12
    n_false_negatives: int = 128
    negatives_from_same_seq_only:bool = False # help="only draw false negatives from same sequence (as opposed to drawing from everywhere)")
    negatives_selection_interval: int = 0 #only draw negative from -x...x around the current index
    normalize: bool = False #normalize before calculating similarities
    temperature: float = 1.0 #temperature parameter dividing similarities

class SyncFunction(torch.autograd.Function):
    '''
    SyncFunction from bolts
    https://github.com/Lightning-Universe/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py
    todo: compare to sync function from the self-supervised cookbook
    '''
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


class CLIPLoss(nn.Module):
    '''CLIP loss'''
    def __init__(self, hparams_loss):
        super().__init__()
        
        self.sigmoid = hparams_loss.sigmoid
        self.tprime = torch.nn.Parameter(torch.tensor(np.log(10)).float())

        if(self.sigmoid):    
            self.b = torch.nn.Parameter(torch.tensor(-10.))
        
    def forward(self,input_predicted,static_encoded,**kwargs):
        #both input shape bs,feat- afterwards world_size*bs,feat

        #for the sigmoid loss there are better solutions than an all_gather
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            input_predicted_dist = SyncFunction.apply(input_predicted)
            static_encoded_dist = SyncFunction.apply(static_encoded)
        else:
            input_predicted_dist = input_predicted
            static_encoded_dist = static_encoded

        #normalize
        input_predicted_dist = F.normalize(input_predicted_dist, p=2.0, dim = -1)
        static_encoded_dist = F.normalize(static_encoded_dist, p=2.0, dim = -1)
        
        sim = (input_predicted_dist @ static_encoded_dist.T)
        loss = 0.
        if(self.sigmoid):
            logits = sim*torch.exp(self.tprime) + self.b
            labels = 2 * torch.eye(sim.shape[0],device=logits.device) - torch.ones(sim.shape[0],device=logits.device)
            loss =  -torch.mean(torch.nn.LogSigmoid()(labels*logits))
        else:
            sim = sim*torch.exp(self.tprime)
            loss = -0.5*torch.mean(torch.diagonal(torch.nn.LogSoftmax(dim=0)(sim)+torch.nn.LogSoftmax(dim=1)(sim)))
        return {"loss": loss}
        
@dataclass
class CLIPLossConfig(SSLLossConfig):
    _target_:str = "clinical_ts.loss.selfsupervised.CLIPLoss"
    loss_type:str = "instance_contrastive_clip"
    target_dim:int = 128 #output dimension of both encoders
    temperature:float = 0.1 #temperature parameter for softmax loss
    sigmoid:bool = False #sigmoid loss from 2303.15343 instead of standard softmax loss


class InfoNCELoss(nn.Module):
    '''InfoNCE loss: accepts both pooled representions (B,E) or token level representations (B,S,E) where overlapping tokens are identified via seq_idx'''
    def __init__(self, hparams_loss):
        super().__init__()
        
        self.t = hparams_loss.temperature
        self.negatives_from_same_seq_only = hparams_loss.negatives_from_same_seq_only
    
    def forward(self,input_predicted,**kwargs):
        #both input shape bs,feat- afterwards world_size*bs,feat

        input_predicted_dist = F.normalize(input_predicted, p=2.0, dim = -1)
        if(len(input_predicted.shape)==2):#pooled representations bs,feat
            sim = (input_predicted_dist[:len(input_predicted_dist)//2] @ input_predicted_dist[len(input_predicted_dist)//2:].T)
            sim = sim / self.t
            loss = -0.5*torch.mean(torch.diagonal(torch.nn.LogSoftmax(dim=0)(sim)+torch.nn.LogSoftmax(dim=1)(sim)))
        elif(len(input_predicted.shape)==3): #token-level representations B,S,E
            bs = input_predicted.shape[0]//2
            seq_len = input_predicted.shape[1]
            seq_idxs = kwargs["seq_idxs"].view(bs,2,3)
            starts_overlap = torch.maximum(seq_idxs[:,0,1], seq_idxs[:,1,1])
            ends_overlap = torch.minimum(seq_idxs[:,0,2], seq_idxs[:,1,2])-1 

            starts_overlap_seq1 = (starts_overlap-seq_idxs[:,0,1])/(seq_idxs[:,0,2]-seq_idxs[:,0,1])*seq_len
            ends_overlap_seq1 = (ends_overlap-seq_idxs[:,0,1])/(seq_idxs[:,0,2]-seq_idxs[:,0,1])*seq_len+1

            #round and check if in bounds
            starts_overlap_seq1 = torch.round(starts_overlap_seq1)
            ends_overlap_seq1 = torch.round(ends_overlap_seq1)
            starts_overlap_seq1 = torch.maximum(torch.zeros_like(starts_overlap),starts_overlap_seq1)
            ends_overlap_seq1 = torch.minimum((seq_len-1)*torch.ones_like(starts_overlap),ends_overlap_seq1)
            
            overlap_seq1 = [[] if starts_overlap_seq1[i]>ends_overlap_seq1[i] else torch.arange(starts_overlap_seq1[i],ends_overlap_seq1[i],dtype=torch.int64,device=input_predicted.device) for i in range(len(starts_overlap))]
            overlap_seq2 = [[] if len(overlap_seq1[i])==0 else torch.round((overlap_seq1[i]/seq_len*(seq_idxs[i,0,2]-seq_idxs[i,0,1])+seq_idxs[i,0,1]-seq_idxs[i,1,1])/(seq_idxs[i,1,2]-seq_idxs[i,1,1])*seq_len).long() for i in range(len(starts_overlap))]
            if(self.negatives_from_same_seq_only):
                overlap_seq1 = [[] if len(s)==0 else s+bs*i if not self.negatives_from_same_seq_only else 0 for i,s in enumerate(overlap_seq1)]
                overlap_seq2 = [[] if len(s)==0 else s+bs*i if not self.negatives_from_same_seq_only else 0 for i,s in enumerate(overlap_seq2)]
                
                loss = 0.
                for i in range(bs):
                    if(len(overlap_seq1[i])==0):
                        continue
                
                    sim = (input_predicted_dist[i] @ input_predicted_dist[i+bs].T)
                    sim = sim / self.t
                    if(len(overlap_seq1[i])>0):
                        loss += -0.5*torch.mean((torch.nn.LogSoftmax(dim=0)(sim)+torch.nn.LogSoftmax(dim=1)(sim))[overlap_seq1[i],overlap_seq2[i]])
            else:
                overlap_seq1 = [x for x in overlap_seq1 if len(x)>0]
                overlap_seq2 = [x for x in overlap_seq2 if len(x)>0]
                
                if(len(overlap_seq1)>0):
                    overlap_seq1 = torch.cat(overlap_seq1)
                    overlap_seq2 = torch.cat(overlap_seq2)
                    sim = (input_predicted_dist[:len(input_predicted_dist)//2].view(bs*seq_len,-1) @ input_predicted_dist[len(input_predicted_dist)//2:].view(bs*seq_len,-1).T)
                    sim = sim / self.t
                    loss = -0.5*torch.mean((torch.nn.LogSoftmax(dim=0)(sim)+torch.nn.LogSoftmax(dim=1)(sim))[overlap_seq1,overlap_seq2]) if(len(overlap_seq1)>0) else 0.
                else:
                    loss = 0.
        return {"loss": loss}
        
@dataclass
class InfoNCELossConfig(SSLLossConfig):
    _target_:str = "clinical_ts.loss.selfsupervised.InfoNCELoss"
    loss_type:str = "instance_contrastive_sequence"
    target_dim:int = 128 #output dimension of both encoders
    temperature:float = 0.1 #temperature parameter for softmax loss
    negatives_from_same_seq_only:bool = False #draw negatives from same sequence only (only applies to token-level contrastive loss a la ts2vec)
