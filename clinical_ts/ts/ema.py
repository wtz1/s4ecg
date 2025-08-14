__all__ = ['EMATimeSeriesEncoder', 'EMATimeSeriesEncoderConfig']

import torch
from torch import nn
from operator import attrgetter
from ..utils.callbacks import ForwardHook

import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Union

from ..template_modules import TimeSeriesEncoder, EMATimeSeriesEncoderBaseConfig

###############################################################################################
# adapted from CAPI only clustering from model.py in https://github.com/facebookresearch/capi/
###############################################################################################
exp_max_values = {
    torch.float16: 0,
    torch.float32: 50,
    torch.float64: 50,
    torch.bfloat16: 50,
}


def stable_exp(M: torch.Tensor) -> torch.Tensor:
    """Compute stable exponential of input tensor.
    
    Args:
        M: Input tensor to compute exponential of
        
    Returns:
        Tensor with stable exponential values
        
    Raises:
        ValueError: If input tensor is empty or contains invalid values
        RuntimeError: If distributed operation fails when initialized
    """
    if M.numel() == 0:
        raise ValueError("Input tensor is empty")
    if torch.isnan(M).any() or torch.isinf(M).any():
        raise ValueError("Input tensor contains NaN or Inf values")
        
    shift = M.max(dim=-2, keepdim=True).values
    
    # Only attempt distributed operation if initialized
    if torch.distributed.is_initialized():
        try:
            torch.distributed.all_reduce(shift, torch.distributed.ReduceOp.MAX)
        except Exception as e:
            raise RuntimeError(f"Distributed operation failed: {e}")
    
    M = M + exp_max_values[M.dtype] - shift
    return M.exp()


def reduced_sum(*args, **kwargs):
    summed = torch.sum(*args, **kwargs)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(summed)
    return summed


@torch.no_grad()
def sinkhorn_knopp(M,  n_iterations: int, eps:float= 1e-8,):
#def sinkhorn_knopp( M: Float[Tensor, "*b n p"],  n_iterations: int, eps: float | int = 1e-8,) -> Float[Tensor, "*b n p"]:
    M = stable_exp(M)
    for _ in range(n_iterations):
        M /= reduced_sum(M, dim=-2, keepdim=True) + eps
        M /= torch.sum(M, dim=-1, keepdim=True) + eps
    return M

class OnlineClustering(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool,  # god why is this bias still here
        n_sk_iter: int,
        target_temp: Union[float, int],
        pred_temp: Union[float, int],
        positionwise_sk: bool = True,
        prefactor_clustering_loss: float = 1.0
    ):
        super().__init__()
        self.out_dim = out_dim
        self.n_sk_iter = n_sk_iter
        self.target_temp = target_temp
        self.pred_temp = pred_temp
        self.positionwise_sk = positionwise_sk
        self.prefactor_clustering_loss = prefactor_clustering_loss
        self.layer = nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.normal_(self.layer.weight, std=1)
        if bias:
            torch.nn.init.zeros_(self.layer.bias)

    #def forward(self, x: Float[Tensor, "*b in_dim"]) -> tuple[Float[Tensor, "*b out_dim"], Float[Tensor, ""]]:
    def forward(self, x):
        x_n = nn.functional.normalize(x, dim=-1, p=2, eps=1e-7)
        logits = self.layer(x_n)
        if not self.positionwise_sk:
            logits = logits.flatten(0, -2)
        assignments = sinkhorn_knopp(logits.detach() / self.target_temp, n_iterations=self.n_sk_iter)
        tgt = assignments.flatten(0, -2).float()
        pred = logits.flatten(0, -2).float()
        loss = -torch.sum(tgt * F.log_softmax(pred / self.pred_temp, dim=-1), dim=-1).mean()
        return assignments.detach(), self.prefactor_clustering_loss*loss

##########################################################################################################
# OnlineKMeans
##########################################################################################################
class OnlineKMeans:
    def __init__(self, n_clusters, learning_rate=0.1):
        """
        Initialize online k-means clustering

        Args:
            n_clusters (int): Number of clusters
            learning_rate (float): Learning rate for centroid updates
        """
        self.n_clusters = n_clusters
        self.learning_rate = learning_rate
        self.centroids = None
        self.counts = None

    def initialize(self, batch):
        """
        Initialize cluster centroids and counts based on the first batch of data

        Args:
            batch (torch.Tensor): First batch of data points [batch_size, dim]
        """
        self.dim = batch.shape[1]
        self.centroids = torch.randn(self.n_clusters, self.dim).to(batch.device)
        self.counts = torch.ones(self.n_clusters).to(batch.device)

    def update(self, batch):
        """
        Update cluster centroids using a batch of data points

        Args:
            batch (torch.Tensor): Batch of data points [batch_size, dim]
        """
        # Initialize centroids and counts if not done already
        if self.centroids is None:
            self.initialize(batch)

        # Find nearest centroid for each point
        distances = torch.cdist(batch, self.centroids)
        closest = torch.argmin(distances, dim=1)

        # Update centroids based on assigned points
        for i in range(self.n_clusters):
            mask = (closest == i)
            if not torch.any(mask):
                continue

            points = batch[mask]

            # Compute the mean of points assigned to this cluster
            centroid_update = points.mean(dim=0)

            # Update centroid using learning rate
            self.centroids[i] = (1 - self.learning_rate) * self.centroids[i] + \
                               self.learning_rate * centroid_update

            # Update counts
            self.counts[i] += len(points)

    def predict(self, X):
        """
        Assign data points to nearest clusters

        Args:
            X (torch.Tensor): Data points to assign [n_samples, dim]

        Returns:
            torch.Tensor: Cluster assignments [n_samples]
        """
        distances = torch.cdist(X, self.centroids)
        return torch.argmin(distances, dim=1)

class EMATimeSeriesEncoder(TimeSeriesEncoder):
    '''A time series encoder consisting of encoder, predictor, head'''
    def __init__(self, hparams_seqenc, hparams_input_shape, static_stats_train, target_dim=None, has_ema=False):
        super().__init__(hparams_seqenc, hparams_input_shape, static_stats_train, target_dim=target_dim, has_ema=has_ema)
        self.decay = hparams_seqenc.ema.decay
        self.update_ema = hparams_seqenc.ema.update_ema
        self.pretrained = hparams_seqenc.ema.pretrained
        self.update_every_k_batches = hparams_seqenc.ema.update_every_k_batches
        self.clustering_algorithm = hparams_seqenc.ema.clustering_algorithm
        self.batch_count = 0

        # Store hook as None initially
        self.hook = None
        self._setup_hook(hparams_seqenc.ema.module_name)

        if(self.clustering_algorithm==0):#online clustering a la capi
            self.clustering = nn.ModuleList([OnlineClustering(in_dim=hparams_seqenc.ema.target_dim if hparams_seqenc.ema.target_dim>0 else hparams_seqenc.pred.model_dim,out_dim=k,bias=hparams_seqenc.ema.capi_bias,n_sk_iter=hparams_seqenc.ema.capi_n_sk_iter,target_temp=hparams_seqenc.ema.capi_target_temp,pred_temp=hparams_seqenc.ema.capi_pred_temp,positionwise_sk
        =hparams_seqenc.ema.capi_positionwise_sk,prefactor_clustering_loss=hparams_seqenc.ema.capi_prefactor_clustering_loss) for k in hparams_seqenc.loss.kmeans_ks])
        else:#k-means a la hubert  
            self.clustering = [OnlineKMeans(n_clusters=k,learning_rate=hparams_seqenc.ema.k_means_learning_rate) for k in hparams_seqenc.loss.kmeans_ks]
        
        #disable loss calculation (and other optional modules)
        self.loss = None
        self.head_ssl = None
        self.quantizer = None
        self.masking = None

    def _setup_hook(self, module_name):
        """Setup forward hook on specified module"""
        if "[" in module_name:
            name = module_name.split('[')[0]
            index = int(module_name.split('[')[1][:-1])
            getter = attrgetter(name)
            hook_module = getter(self)[index]
        else:
            getter = attrgetter(module_name)
            hook_module = getter(self)
        
        # Remove existing hook if present
        self._cleanup_hook()
        # Create new hook
        self.hook = ForwardHook(hook_module, store_output=True)

    def _cleanup_hook(self):
        """Cleanup existing hook if present"""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def __del__(self):
        """Ensure hook is cleaned up when object is deleted"""
        self._cleanup_hook()

    def initialize_params(self, ts_encoder):
        '''supposed to be called before fit_start'''
        with torch.no_grad():
            ema_state_dict = self.state_dict()

            if(self.pretrained==""):#use weights from parent module
                other_state_dict = ts_encoder.state_dict()
            else:
                checkpoint = torch.load(self.pretrained, map_location=lambda storage, loc: storage,)
                other_state_dict = checkpoint["state_dict"]
            
            for key in ema_state_dict.keys():
                other_key = key[len("ema."):]
                if(other_key in other_state_dict.keys()):
                    ema_state_dict[key].data.copy_(other_state_dict[other_key].data)

            self.load_state_dict(ema_state_dict)

    def update_params(self, ts_encoder):
        '''supposed to be called after every batch'''
        if(self.update_ema):
            self.batch_count = self.batch_count +1
            if(self.batch_count % self.update_every_k_batches ==0):
                with torch.no_grad():
                    ema_state_dict = self.state_dict()
                    other_state_dict = ts_encoder.state_dict()

                    for key in ema_state_dict.keys():
                        other_key = key[len("ema."):]#strip ema. from key name
                        if(other_key in other_state_dict.keys()):
                            ema_state_dict[key].data.copy_(self.decay * ema_state_dict[key].data + (1 - self.decay) * other_state_dict[other_key].data)

                    self.load_state_dict(ema_state_dict)
                
    
    def forward_features(self, **kwargs):
        #just run the model to populate the forward hook
        with torch.no_grad():
            self.forward(**kwargs)
        
        #grab features from the hook
        features = self.hook.stored
        if(isinstance(features,dict)):
            features = features["seq"]
        if(isinstance(features,tuple)):#some layers like lstms, s4 etc might return tuples
            features = features[0] #just pick the first
        if(len(self.transpose_axes_after_hook)>0):
            features = features.permute(*self.transpose_axes_after_hook)

        N = features.shape[1]
        #remove seq axis
        features = features.reshape(-1,features.shape[2])

        #cluster them
        if(self.clustering_algorithm==0):#capi-like online clustering
            loss_clustering = 0 
            cluster_assignments = []
            for c in self.clustering:
                ca, loss = c(features)
                cluster_assignments.append(ca.view(-1,N,ca.shape[-1]))#revive seq axis
                loss_clustering = loss_clustering + loss
            return {"ema_soft_cluster_assignments":cluster_assignments, "loss_ema_clustering": loss_clustering}
        elif(self.clustering_algorithm==1):#k-means
            cluster_labels = []
            cluster_centroids = []
            for c in self.clustering:
                c.update(features)
                cluster_labels.append(c.predict(features).view(-1,N))#revive seq axis
                cluster_centroids.append(c.centroids)
            return {"ema_cluster_labels":cluster_labels, "ema_cluster_centroids":cluster_centroids}

@dataclass
class EMATimeSeriesEncoderConfig(EMATimeSeriesEncoderBaseConfig):
    _target_:str = "clinical_ts.ts.ema.EMATimeSeriesEncoder"
    #EMA parameters
    decay:float = 0.99 #decay parameter for EMA
    pretrained:str = "" #path to pretrained model "" will default to using parent module for initialization
    update_ema:bool = True #avoid EMA update (e.g. when using a pretrained model as in HuBERT)
    update_every_k_batches:int = 5

    #targets
    module_name:str = "predictor" #also supports [i] for module lists such as predictor.predictor.s4_layers[1]
    transpose_axes_after_hook:List[int]=field(default_factory=lambda: [])#can pass a permuation order e.g. [0,2,1] for intermediate s4 layers
    target_dim:int = 0 #dimensionality of the target; default=0 assumes predictor.model_dim

    #clustering
    clustering_algorithm:int = 0#0: capi 1: online k-means

    #k-means
    k_means_learning_rate:float = 0.1
    
    #capi
    capi_prefactor_clustering_loss:float = 0.05
    capi_target_temp:float = 0.06
    capi_pred_temp:float = 0.12
    capi_n_sk_iter:int = 3
    capi_bias:bool = True
    capi_positionwise_sk:bool = False
