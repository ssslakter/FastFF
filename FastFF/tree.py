# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_tree.ipynb.

# %% auto 0
__all__ = ['draw_line', 'idx_to_tree_path', 'get_mask', 'get_leaves', 'plot_contour', 'ContourCB']

# %% ../nbs/01_tree.ipynb 3
from fastai.vision.all import *
import matplotlib.colors as mcolors

# %% ../nbs/01_tree.ipynb 5
def draw_line(x,y, ax = None, figsize=(5,3), **kwargs):
    if not ax: ax = subplots(figsize=figsize)[1][0]
    ax.plot(x,y, **kwargs)

# %% ../nbs/01_tree.ipynb 13
def idx_to_tree_path(idx: int):   # node index
    '''get turns and nodes of binary tree to reach `idx` node'''
    turns = list(map(int,[*bin(idx+1)[3:]]))
    nodes = reduce(lambda l,i: l+[l[-1]*2+2 if i else l[-1]*2+1], turns, [0])
    return turns, nodes

# %% ../nbs/01_tree.ipynb 15
def get_mask(xb: Tensor,    # input tensor of shape (b_size, n_inp)
             idx: int,      # index of node
             ws: Tensor,    # model weights of shape (n_nodes, n_inp)
             bs: Tensor):   # model biases of shape (n_nodes,)
    '''get only points that end up at `idx` node'''
    assert idx <= ws.shape[0]*2, 'node in not in tree'
    xb = to_device(xb, ws.device)
    mask = torch.ones(xb.shape[0], device=ws.device).bool()
    ts, ns = idx_to_tree_path(idx)
    for t,v in zip(ts,ns[:-1]):
        m = (xb@ws[v]+bs[v]>0)
        if not t: m = ~m
        mask = mask & m
    return mask

# %% ../nbs/01_tree.ipynb 20
def get_leaves(module, x_lim=(-10,10), n= 500):
    module.eval(); module.skip_out = True
    x_lin = torch.linspace(*x_lim, n)
    X, Y = torch.meshgrid(x_lin, x_lin, indexing='xy')
    xs = to_device(torch.stack([X,Y],dim=2).view(-1,X.shape[0],2), module.w1s.device)
    with torch.no_grad(): 
        Z = torch.stack([module(b) for b in xs]).view(X.shape)
    module.skip_out = False
    return X,Y,Z
    
def plot_contour(X,Y,Z, ax=None, cmap = 'plasma', figsize=(6,4), **kwargs):
    clev = torch.linspace(Z.min(),Z.max(),50)
    if not ax: ax = subplots(figsize=figsize)[1][0]
    if not isinstance(cmap,mcolors.Colormap): cmap = plt.get_cmap(cmap)
    ax.contourf(*to_np((X,Y,Z)), clev, cmap=cmap, **kwargs)

# %% ../nbs/01_tree.ipynb 22
class ContourCB(Callback):
    '''After each epoch calls `get_leaves` on all input space'''
    def __init__(self, module, x_lim=(-50,50), n=500): store_attr()
        
    def before_fit(self): self.z = []
    
    def after_epoch(self): 
        if not self.training:
            X,Y,Z = get_leaves(self.module, self.x_lim,self.n)
            if not hasattrs(self,('X','Y')): self.X,self.Y=X,Y
            self.z.append(Z)

    @delegates(plot_contour)
    def show(self, idx=-1, **kwargs):
        plot_contour(self.X,self.Y,self.z[idx],**kwargs)
    
