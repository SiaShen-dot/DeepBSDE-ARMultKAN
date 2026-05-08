import torch
import torch.nn as nn
import numpy as np
from .RKANLayer import RKANLayer
from .spline import curve2coef

class RMultKAN(nn.Module):
    '''
    RMultKAN: Residual Multiplicative Kolmogorov-Arnold Network
    Simplified for Physics-Informed Learning (PDE Solving)
    '''
    def __init__(self, width=None, grid=3, k=3, mult_arity=2, noise_scale=0.3, 
                 scale_base_mu=0.0, scale_base_sigma=1.0, base_fun='silu', 
                 affine_trainable=False, grid_eps=0.02, grid_range=[-1, 1], 
                 sp_trainable=True, sb_trainable=True, seed=42, 
                 device='cpu'):
        '''
        Args:
            width : list
                Network architecture. e.g. [2, 5, 5, 1]
                Or with mult nodes: [2, [5, 2], 5, 1] (5 sum nodes, 2 mult nodes)
            grid : int
                Initial grid size (G)
            k : int
                Spline order (k)
            mult_arity : int
                Arity for multiplication nodes (default 2)
        '''
        super(RMultKAN, self).__init__()

        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.width = width
        self.depth = len(width) - 1
        self.grid = grid
        
        # [修改点 1] 使用内部变量 _grid_eps 存储，避免初始化冲突
        self._grid_eps = grid_eps 
        
        self.k = k
        self.base_fun_name = base_fun
        self.device = device
        
        # 乘法节点配置
        if isinstance(mult_arity, int):
            self.mult_homo = True 
        else:
            self.mult_homo = False 
        self.mult_arity = mult_arity

        # 激活函数
        if base_fun == 'silu':
            base_fun_act = torch.nn.SiLU()
        elif base_fun == 'identity':
            base_fun_act = torch.nn.Identity()
        elif base_fun == 'zero':
            base_fun_act = lambda x: x*0.
        else:
            base_fun_act = torch.nn.SiLU()

        # --- 初始化网络层 ---
        self.act_fun = []
        width_in = self.width_in
        width_out = self.width_out

        for l in range(self.depth):
            sp_batch = RKANLayer(
                in_dim=width_in[l], 
                out_dim=width_out[l+1], 
                num=grid, 
                k=k, 
                noise_scale=noise_scale, 
                scale_base_mu=scale_base_mu, 
                scale_base_sigma=scale_base_sigma, 
                scale_sp=1., 
                base_fun=base_fun_act, 
                grid_eps=grid_eps, # 初始化时直接使用传入的参数
                grid_range=grid_range, 
                sp_trainable=sp_trainable, 
                sb_trainable=sb_trainable,
            )
            self.act_fun.append(sp_batch)
        
        self.act_fun = nn.ModuleList(self.act_fun)

        self.to(device)
        self.input_id = torch.arange(self.width_in[0])
        
        # 缓存激活值用于网格更新
        self.acts = None 

    # [修改点 2] 添加 property 装饰器，实现 grid_eps 的联动更新
    @property
    def grid_eps(self):
        return self._grid_eps

    @grid_eps.setter
    def grid_eps(self, value):
        # 当外部执行 model.grid_eps = value 时，自动运行此函数
        self._grid_eps = value
        # 遍历所有层，同步更新
        if hasattr(self, 'act_fun'):
            for layer in self.act_fun:
                layer.grid_eps = value

    def to(self, device):
        super(RMultKAN, self).to(device)
        self.device = device
        for rkanlayer in self.act_fun:
            rkanlayer.to(device)
        return self
    
    @property
    def width_in(self):
        width = self.width
        # 统一格式：如果是 int 则转为 [int, 0]
        width_corrected = [([w, 0] if isinstance(w, int) else w) for w in width]
        return [w[0] + w[1] for w in width_corrected]
        
    @property
    def width_out(self):
        width = self.width
        width_corrected = [([w, 0] if isinstance(w, int) else w) for w in width]
        if self.mult_homo:
            return [w[0] + self.mult_arity * w[1] for w in width_corrected]
        else:
            return [w[0] + int(np.sum(self.mult_arity[l])) for l, w in enumerate(width_corrected)]

    def forward(self, x):
        '''
        Args:
            x : 2D torch.tensor [batch, dim]
        '''
        # 确保输入维度匹配
        if x.shape[1] != self.width_in[0]:
             x = x[:, self.input_id.long()]

        # 缓存输入用于第一层的 grid update
        self.acts = [] 
        self.acts.append(x)

        for l in range(self.depth):
            
            # 1. KAN Layer 计算 (Spline + Base)
            x = self.act_fun[l](x)
            
            # 2. 处理乘法节点 (Multiplication Nodes)
            w_next = self.width[l+1]
            if isinstance(w_next, int): w_next = [w_next, 0]
            
            dim_sum = w_next[0]
            dim_mult = w_next[1]
            
            # 如果这一层有乘法节点
            if dim_mult > 0:
                if self.mult_homo:
                    # 并行乘法逻辑
                    for i in range(self.mult_arity-1):
                        if i == 0:
                            x_mult = x[:,dim_sum::self.mult_arity] * x[:,dim_sum+1::self.mult_arity]
                        else:
                            x_mult = x_mult * x[:,dim_sum+i+1::self.mult_arity]
                else:
                    # 异构乘法逻辑 (for loop)
                    for j in range(dim_mult):
                        acml_id = dim_sum + np.sum(self.mult_arity[l+1][:j])
                        for i in range(self.mult_arity[l+1][j]-1):
                            if i == 0:
                                x_mult_j = x[:,[acml_id]] * x[:,[acml_id+1]]
                            else:
                                x_mult_j = x_mult_j * x[:,[acml_id+i+1]]
                        if j == 0:
                            x_mult = x_mult_j
                        else:
                            x_mult = torch.cat([x_mult, x_mult_j], dim=1)
                
                # 将加法节点和乘法节点拼接
                x = torch.cat([x[:,:dim_sum], x_mult], dim=1)
            
            # 缓存当前层的激活值，供下一层的 update_grid 使用
            self.acts.append(x.detach())
            
        return x

    def update_grid(self, x):
        '''
        Adaptive Grid Update (AGU) 核心函数
        根据输入样本 x 更新 B-spline 的网格节点位置
        '''
        # 先跑一次前向传播，更新 self.acts
        self.forward(x) 
        
        for l in range(self.depth):
            # 调用 RKANLayer 内部的 grid update
            self.act_fun[l].update_grid_from_samples(self.acts[l])