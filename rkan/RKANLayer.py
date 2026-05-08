import torch
import torch.nn as nn
import numpy as np
from .spline import *

class RKANLayer(nn.Module):
    """
    RKANLayer class (Simplified for PDE Solving)
    """
    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.5, 
                 scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0, 
                 base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], 
                 sp_trainable=True, sb_trainable=True, device='cpu'):
        
        super(RKANLayer, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k
        self.grid_eps = grid_eps
        self.device = device

        # Residual Connection: Match dimensions if necessary
        if in_dim != out_dim:
            self.match_dim = torch.nn.Linear(in_dim, out_dim)
        else:
            self.match_dim = None

        # Initialize Grid
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None,:].expand(self.in_dim, num+1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        
        # Initialize Coefficients with Noise
        noises = (torch.rand(self.num+1, self.in_dim, self.out_dim) - 1/2) * noise_scale / num
        self.coef = torch.nn.Parameter(curve2coef(self.grid[:,k:-k].permute(1,0), noises, self.grid, k))
        
        # Initialize Scaling Factors
        self.scale_base = torch.nn.Parameter(scale_base_mu * 1 / np.sqrt(in_dim) + \
                         scale_base_sigma * (torch.rand(in_dim, out_dim)*2-1) * 1/np.sqrt(in_dim)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp).requires_grad_(sp_trainable)
        
        self.base_fun = base_fun
        self.to(device)
        
    def to(self, device):
        super(RKANLayer, self).to(device)
        self.device = device    
        return self

    def forward(self, x):
        batch = x.shape[0]
        
        # 1. Base Function (SiLU)
        base = self.base_fun(x) # (batch, in_dim)
        
        # 2. B-Spline Function
        # preacts: inputs to splines, saved for grid update
        preacts = x[:,None,:].clone().expand(batch, self.out_dim, self.in_dim)
        y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k) # (batch, in_dim, out_dim)
        
        postspline = y.clone().permute(0,2,1)
            
        # 3. Apply Scaling
        y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_sp[None,:,:] * y
        
        postacts = y.clone().permute(0,2,1)
            
        # 4. Sum over input dimensions
        y = torch.sum(y, dim=1) # (batch, out_dim)

        # 5. Residual Connection
        if self.match_dim is not None:
            x_residual = self.match_dim(x)
        else:
            x_residual = x

        y_residual = y + x_residual

        return y_residual

    def update_grid_from_samples(self, x):
        '''
        Adaptive Grid Update (AGU) Logic
        '''
        batch = x.shape[0]
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        num_interval = self.grid.shape[1] - 1 - 2*self.k
        
        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1,0)
            h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]])/num_interval
            grid_uniform = grid_adaptive[:,[0]] + h * torch.arange(num_interval+1,)[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid
        
        grid = get_grid(num_interval)
        self.grid.data = extend_grid(grid, k_extend=self.k)
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)