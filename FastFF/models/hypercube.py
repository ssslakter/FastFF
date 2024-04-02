# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/08_hypercube.ipynb.

# %% auto 0
__all__ = ['HyperCubeMoE']

# %% ../../nbs/08_hypercube.ipynb 1
import torch
import torch.nn as nn, torch.nn.functional as F 
from .moe import *
from fastcore.all import *

# %% ../../nbs/08_hypercube.ipynb 3
class HyperCubeMoE(nn.Module):
    '''Mixture of experts network'''
    def __init__(self, in_dim, out_dim, gate_dim=2, top_k=2, hidden_dim=128, act=nn.ReLU, save_probs=True):
        super().__init__()
        store_attr()
        self.gate = lin(in_dim, gate_dim, act=act, bias=False)
        self.experts = Experts(mlp(in_dim,out_dim, hidden_dim, act=act) for _ in range(2**gate_dim))
        self.mask = binary(torch.arange(0,2**gate_dim), gate_dim)*2-1.
    
    def forward(self,x):
        logits = self.gate(x)
        # probs = F.logsigmoid(logits[:,None,:]*self.mask).sum(-1).exp()
        probs = F.softmax((logits[:,None,:]*self.mask).sum(-1),1)
        if self.save_probs: self.probs = probs
        probs, selected_exps = torch.topk(probs, self.top_k, dim=-1)
        probs /= probs.sum(dim=-1, keepdim=True)
        return self.experts(x, probs, selected_exps)
