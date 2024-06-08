from math import sqrt

import torch
import torch.nn as nn
from torch.nn import functional as F


class MoE(nn.Module):
    def __init__(self, in_dim, out_dim, n_parallel, n_exp, top_k=1, act=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_parallel = n_parallel  # n_parallel
        self.n_exp = n_exp  # d
        self.top_k = top_k

        def uniform(shape, scale):
            return nn.Parameter(torch.empty(shape).uniform_(-scale, scale))
        self.w1 = uniform((in_dim, self.n_parallel, self.n_exp), scale=1 / sqrt(in_dim))
        self.w2 = uniform((self.n_parallel, self.n_exp, out_dim), scale=1 / sqrt(self.n_parallel * self.n_exp))
        self.act = act()

    def forward(self, x: torch.Tensor):
        s = x.shape
        x = x.view(-1,self.in_dim)
        x = torch.matmul(x, self.w1.view(self.in_dim, -1))
        # make nonmaximal activations zero
        x = x.view(x.shape[0], self.n_parallel, self.n_exp)

        top_values, top_indices = x.topk(self.top_k, dim=2)
        z = torch.zeros_like(x)
        z.scatter_(2, top_indices, top_values)
        # z = self.act(z)
        z = torch.matmul(z.view(z.shape[0], -1), self.w2.view(self.n_parallel * self.n_exp, self.out_dim))
        return z.view(*s[:-1],self.out_dim)
