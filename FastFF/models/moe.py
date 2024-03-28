# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/05_moe.ipynb.

# %% auto 0
__all__ = ['lin', 'mlp', 'Experts', 'MoE', 'FFF', 'InitFFF']

# %% ../../nbs/05_moe.ipynb 2
import math, torch, torch.nn.functional as F
from torch import nn
from fastai.vision.all import *

# %% ../../nbs/05_moe.ipynb 3
def lin(in_dim, out_dim, act=nn.ReLU, bias=True):
    '''Linear layer followed by activation'''
    if act is None: act = nn.Identity
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias), act())

def mlp(in_dim, out_dim, hidden_dim=128, n_hidden=1, act=nn.ReLU, bias=True):
    '''Multilayer perceptron with several hidden layers'''
    if n_hidden==0: return lin(in_dim, out_dim, act, bias)
    res = nn.Sequential(*lin(in_dim, hidden_dim, act, bias))
    for _ in range(n_hidden-1): res+= lin(hidden_dim, hidden_dim, act, bias)
    res += lin(hidden_dim, out_dim, act, bias)
    return res

# %% ../../nbs/05_moe.ipynb 4
# benchmark comparing nn.ModuleList and nn.Conv1d https://discuss.pytorch.org/t/parallel-execution-of-modules-in-nn-modulelist/43940/11
class Experts(nn.ModuleList):
    """A class representing a collection of experts. Will compute weighted sum of results of topk experts depending on `selected_exps`"""
    
    def forward(self, x, routing_ws, selected_exps):
        mask = F.one_hot(selected_exps, num_classes=len(self)).permute(2, 1, 0)
        for i in range(len(self)):
            idx, top_x = torch.where(mask[i])
            if top_x.shape[0] == 0: continue
            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            res = self[i](x[top_x_list]) * routing_ws[top_x_list, idx.tolist(), None]
            if 'out' not in locals(): out = torch.zeros((x.shape[0],*res.shape[1:]), device=x.device)
            out.index_add_(0, top_x, res)
        return out


class MoE(nn.Module):
    '''Mixture of experts network'''
    def __init__(self, in_dim, out_dim, n_experts=4, top_k=4, hidden_dim=128, act=nn.ReLU, save_probs=True):
        super().__init__()
        store_attr()
        self.gate = lin(in_dim, n_experts, act=act, bias=False)
        self.experts = Experts(mlp(in_dim,out_dim, hidden_dim, act=act) for _ in range(n_experts))
    
    def forward(self,x):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=1)
        if self.save_probs: self.probs = probs
        probs, selected_exps = torch.topk(probs, self.top_k, dim=-1)
        probs /= probs.sum(dim=-1, keepdim=True)
        return self.experts(x, probs, selected_exps)

# %% ../../nbs/05_moe.ipynb 5
class FFF(MoE):
    def __init__(self, in_dim, out_dim, depth=2, top_k=4, hidden_dim = 128, act=nn.ReLU, save_probs=True):
        '''FFF which computes leaves probability distribution during forward'''
        store_attr()
        self.n_leaves = 2**depth
        super().__init__(in_dim, out_dim, self.n_leaves, top_k, hidden_dim, act, save_probs)
        # override gate to have size 1 less
        self.gate = lin(in_dim, self.n_leaves-1, act=act, bias=False)
    
    def forward(self, x):
        bs = x.shape[0]
        logits = self.gate(x)
        logprobs = F.logsigmoid(torch.stack([-logits, logits],dim=2))     # (bs, n_leaves-1, 2)
        probs = torch.zeros([bs,self.n_leaves], device=x.device)     # (bs, n_leaves)
        for d in range(self.depth):
            mask = logprobs[:, 2**d-1 : 2**(d+1)-1].view(bs,-1, 1)        # (bs, 2*2**d, 1)
            probs = probs.view(bs, 2**(d+1), -1) + mask         # (bs, 2**(d+1), n_leaves//2**(d+1) )
        probs = torch.exp(probs).view(bs, -1)
        if self.save_probs: self.probs = probs.detach()
        routing_weights, selected_exps = torch.topk(probs, self.top_k, dim=-1)
        return self.experts(x, routing_weights, selected_exps)

# %% ../../nbs/05_moe.ipynb 6
class InitFFF(MoE):
    '''FFF which uses precomputed matrix for leaves distribution'''
    def __init__(self, in_dim, out_dim, depth=2, top_k=4, hidden_dim = 128, act=nn.ReLU, save_probs=True):
        store_attr()
        self.n_leaves = 2**depth
        self.tree = self.init_tree_()
        super().__init__(in_dim, out_dim, self.n_leaves, top_k, hidden_dim, act, save_probs)
        # override gate to have size 1 less
        self.gate = lin(in_dim, self.n_leaves-1, act=act, bias=False)
    
    def init_tree_(self):
        masks = [torch.eye(self.n_leaves)]
        for _ in range(self.depth-1): masks.append(masks[-1].view(-1, 2, self.n_leaves).sum(1))
        return torch.cat(list(reversed(masks)),dim=0)   # (2*n_leaves-2, n_leaves)
    
    def forward(self, x):
        bs = x.shape[0]
        logits = self.gate(x)
        logprobs = F.logsigmoid(torch.stack([-logits, logits],dim=2).view(bs,-1))     # (bs, 2*n_leaves-2)
        probs = torch.exp(torch.matmul(logprobs, self.tree))
        if self.save_probs: self.probs = probs.detach()
        routing_weights, selected_exps = torch.topk(probs, self.top_k, dim=-1)
        return self.experts(x, routing_weights, selected_exps)
