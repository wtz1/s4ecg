__all__ = ['MultiCodebookQuantizer', 'MultiCodebookQuantizerConfig']

import torch
from torch import nn, einsum
import torch.nn.functional as F
from dataclasses import dataclass
import math
from ..utils.callbacks import cos_anneal

from ..template_modules import QuantizerBase, QuantizerBaseConfig



@dataclass
class MultiCodebookQuantizerConfig(QuantizerBaseConfig):
    _target_:str = "clinical_ts.quantizer.multi.MultiCodebookQuantizer"
    quantizer:str = "gumbel"#gumbel or vqvae
    straight_through:bool = True #for gumbel, vqvae is always straight_through

    ramp_kld_scale_num_steps:int = 5000
    ramp_kld_scale_end:float = 5e-4
    decay_temperature_num_steps:int = 150000
    decay_temperature_start:float = 1
    decay_temperature_end:float = 1./16.
    
class MultiCodebookQuantizer(QuantizerBase):
    def __init__(self, hparams_quantizer, hparams_input_shape):
        super().__init__(hparams_quantizer, hparams_input_shape)
        self.quantizers = nn.ModuleList([
            GumbelQuantize(hparams_input_shape.channels, hparams_quantizer.vocab_size, hparams_quantizer.embedding_dim, straight_through=hparams_quantizer.straight_through) if hparams_quantizer.quantizer=="gumbel" else VQVAEQuantize(hparams_input_shape.channels, hparams_quantizer.vocab_size, hparams_quantizer.embedding_dim) for _ in range(hparams_quantizer.num_codebooks)
            ]
            )
        self.proj = nn.Linear(hparams_quantizer.num_codebooks*hparams_quantizer.embedding_dim,hparams_quantizer.target_dim) if hparams_quantizer.target_dim > 0 else nn.Identity()

        #These ramps/decays follow DALL-E Appendix A.2 Training https://arxiv.org/abs/2102.12092
        self.ramp_kld_scale_num_steps = hparams_quantizer.ramp_kld_scale_num_steps 
        self.ramp_kld_scale_end = hparams_quantizer.ramp_kld_scale_end 
        self.decay_temperature_num_steps = hparams_quantizer.decay_temperature_num_steps
        self.decay_temperature_start = hparams_quantizer.decay_temperature_start
        self.decay_temperature_end = hparams_quantizer.decay_temperature_end
        
    def forward(self, **kwargs):
        output_seq = []
        output_soh = []
        #output_ind = []
        loss = 0
        for q in self.quantizers:#quantizers assume batch, features, timesteps
            res = q(kwargs["seq"].transpose(1,2))
            loss += res["diff"]
            
            output_seq.append(res["z_q"])
            if("soft_one_hot" in res.keys()):
                output_soh.append(res["soft_one_hot"])
            
        
        output_seq = torch.cat(output_seq,dim=1).transpose(1,2) #output_seq is batch, timesteps, num_codebooks*embedding_dim
        if(len(output_soh)>0):
            output_soh = torch.cat(output_soh,dim=1).transpose(1,2)
        #output_ind = torch.cat(output_ind,dim=1).transpose(1,2) #output_ind is batch, timesteps,num_codebooks
        
        res={"seq":self.proj(output_seq), "loss_quantizer":loss}
        if(len(output_soh)>0):
            res["soft_one_hot"]=output_soh
        return res

    def update_hyperparams(self, step):
        # The KL weight β is increased from 0 to 6.6 over the first 5000 updates
        # "We divide the overall loss by 256 × 256 × 3, so that the weight of the KL term
        # becomes β/192, where β is the KL weight."
        # TODO: OpenAI uses 6.6/192 but kinda tricky to do the conversion here... about 5e-4 works for this repo so far... :\
        kld_scale = cos_anneal(0, self.ramp_kld_scale_num_steps, 0.0, self.ramp_kld_scale_end, step)
        # The relaxation temperature τ is annealed from 1 to 1/16 over the first 150,000 updates.
        temperature = cos_anneal(0, self.decay_temperature_num_steps, self.decay_temperature_start, self.decay_temperature_end, step)
        kwargs = {"temperature":temperature, "kld_scale":kld_scale}
        for m in self.quantizers:
            m.update_hyperparams(**kwargs)

#from https://github.com/karpathy/deep-vector-quantization/blob/main/dvq/model/quantize.py
class GumbelQuantize(nn.Module):
    """
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144

    num_hiddens: feature dimension of the incoming data
    n_embed: number of embedding tokens/clusters
    embedding_dim: embedding dimension
    straight_through: whether to use straight through estimator
    data_dim: dimensionality of the input data 1d/2d

    temperature and kld_scale can be adjusted via corresponding attributes
    """
    def __init__(self, num_hiddens, n_embed, embedding_dim, straight_through=False, data_dim=1):
        super().__init__()
        self.data_dim = data_dim

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = 1.0
        self.kld_scale = 5e-4

        self.proj = nn.Conv1d(num_hiddens, n_embed, 1) if data_dim==1 else nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

    def forward(self, z):
        '''
        input:
        z: input data with shape (batch, num_hiddens, height, width) for data_dim=2 (input batch, num_hiddens, timesteps) for data_dim=1
        output:
        z_q: weighted sum of embedding vectors (batch, embedding_dim, height, width) or (batch, embedding_dim, timesteps) approaches quantized value in the limit of temperature goes to zero
        diff: kl divergence to the uniform prior (to be added to the loss)
        ind: picked indices (argmax) of shape b h w/ b t (integers between 0 and n_embed)
        '''
        # force hard = True when we are in eval mode, as we must quantize
        hard = self.straight_through if self.training else True

        logits = self.proj(z) # b n h w/ b n t (n=n_embed)
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard) # b n h w/ b n t probability distribution (dim=1)
    
        if(self.data_dim == 1):
            z_q = einsum('b n t, n d -> b d t', soft_one_hot, self.embed.weight) #b d t weighted sum of embedding vectors (according to prob distribution)
        elif(self.data_dim == 2):
            z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight) #b d h w

        # + kl divergence to the prior loss  (uniform 1/n_embed)
        qy = F.softmax(logits, dim=1)
        diff = self.kld_scale * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()    
        #ind = soft_one_hot.argmax(dim=1)
        return {"z_q":z_q, "diff": diff, "soft_one_hot":soft_one_hot} #could also return ind if desired
    
    def update_hyperparams(self, **kwargs):
        self.temperature = kwargs["temperature"]
        self.kld_scale = kwargs["kld_scale"]

#from https://github.com/karpathy/deep-vector-quantization/blob/main/dvq/model/quantize.py
class VQVAEQuantize(nn.Module):
    """
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937

    Follows the original DeepMind implementation
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
    """
    def __init__(self, num_hiddens, n_embed, embedding_dim, data_dim=1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.kld_scale = 10.

        self.proj = nn.Conv1d(num_hiddens, embedding_dim, 1) if data_dim==1 else nn.Conv2d(num_hiddens, embedding_dim, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.data_dim = data_dim
        #self.register_buffer('data_initialized', torch.zeros(1))

    def forward(self, z):
        if(self.data_dim==1):
            B, C, T = z.size()
        elif(self.data_dim==2):
            B, C, H, W = z.size()

        # project and flatten out space, so (B, C, H, W) -> (B*H*W, C)
        z_e = self.proj(z)
        z_e = z_e.permute(0, 2, 1) if self.data_dim==1 else z_e.permute(0, 2, 3, 1) # make (B, H, W, C) or (B, T, C)
        flatten = z_e.reshape(-1, self.embedding_dim) #(-1, C)

        # DeepMind def does not do this but I find I have to... ;\
        #if self.training and self.data_initialized.item() == 0:
        #    print('running kmeans!!') # data driven initialization for the embeddings
        #    rp = torch.randperm(flatten.size(0))
        #    kd = kmeans2(flatten[rp[:20000]].data.cpu().numpy(), self.n_embed, minit='points')
        #    self.embed.weight.data.copy_(torch.from_numpy(kd[0]))
        #    self.data_initialized.fill_(1)
        #    # TODO: this won't work in multi-GPU setups

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1, keepdim=True).t()
        )
        _, ind = (-dist).max(1)

        ind = ind.view(B, T) if self.data_dim==1 else ind.view(B, H, W)

        # vector quantization cost that trains the embedding vectors
        z_q = self.embed_code(ind) # (B, H, W, C)
        commitment_cost = 0.25
        diff = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
        diff *= self.kld_scale

        z_q = z_e + (z_q - z_e).detach() # noop in forward pass, straight-through gradient estimator in backward pass
        z_q = z_q.permute(0, 3, 1, 2) if self.data_dim==2 else z_q.permute(0, 2, 1) # stack encodings into channels again: (B, C, H, W)
        return {"z_q":z_q, "diff":diff, "ind":ind}
    
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)
    
    def update_hyperparams(self, **kwargs):
        self.kld_scale = kwargs["kld_scale"]
