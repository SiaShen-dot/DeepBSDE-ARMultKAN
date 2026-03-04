import numpy as np
import torch
# ===========================
# 方程定义 (PyTorch Version)
# ===========================

class Equation(object):
    def __init__(self, config):
        self.dim = config.dim
        self.total_time = config.total_time
        self.num_time_interval = config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None
        self.device = config.device

    def sample(self, num_sample):
        raise NotImplementedError

    def f_torch(self, t, x, y, z):
        raise NotImplementedError

    def g_torch(self, t, x):
        raise NotImplementedError

class HJBLQ(Equation):
    """
    HJB Equation for Linear-Quadratic Gaussian Control.
    Ref: Han et al., 2018 (https://www.pnas.org/doi/10.1073/pnas.1718942115)
    """
    def __init__(self, config):
        super(HJBLQ, self).__init__(config)
        self.x_init = torch.zeros(self.dim, device=self.device)
        self.sigma = np.sqrt(2.0)
        self.lambd = 1.0

        # [手动添加] 参考解 (Reference Solution)
        # 来源于 PNAS 论文图表或高精度 MC 模拟 (针对 T=1, lambda=1, 100D)
        self.y_init = 4.5901

    def sample(self, num_sample):
        # 使用 PyTorch 生成随机数
        dw_sample = torch.randn(num_sample, self.dim, self.num_time_interval, device=self.device) * self.sqrt_delta_t
        x_sample = torch.zeros(num_sample, self.dim, self.num_time_interval + 1, device=self.device)
        
        # 初始化 x_0
        x_sample[:, :, 0] = self.x_init.unsqueeze(0).expand(num_sample, self.dim)
        
        # 前向 Euler-Maruyama 积分生成路径
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
            
        return dw_sample, x_sample

    def f_torch(self, t, x, y, z):
        # Generator: -lambda * |z|^2 / 2
        # z shape: [batch, dim]
        return -self.lambd * torch.sum(z**2, dim=1, keepdim=True) / 2

    def g_torch(self, t, x):
        # Terminal: log((1 + |x|^2) / 2)
        return torch.log((1 + torch.sum(x**2, dim=1, keepdim=True)) / 2)

import numpy as np
import torch

class PricingDefaultRisk(Equation):
    """
    Nonlinear Black-Scholes equation with default risk in PNAS paper
    Ref: Han et al., 2018 (https://www.pnas.org/doi/10.1073/pnas.1718942115)
    """
    def __init__(self, config):
        super(PricingDefaultRisk, self).__init__(config)
        # --- 论文参数设置 [cite: 138] ---
        self.x_init = torch.ones(self.dim, device=self.device) * 100.0
        self.sigma = 0.2
        self.rate = 0.02    # interest rate R
        self.delta = 2.0 / 3
        self.gammah = 0.2
        self.gammal = 0.02
        self.mu_bar = 0.02
        self.vh = 50.0
        self.vl = 70.0
        # Slope for the piecewise linear function Q(y)
        self.slope = (self.gammah - self.gammal) / (self.vh - self.vl)
        self.device = config.device # 确保这里使用了 config 中的 device

        # 参考解 (Reference Solution)
        # 来源: PNAS 论文 Pg. 8506 [cite: 126]
        # "approximate computed by multilevel Picard method: 57.30"
        self.y_init = 57.30

    def sample(self, num_sample):
        """
        生成 SDE 路径: dX_t = mu_bar * X_t * dt + sigma * X_t * dW_t
        使用 PyTorch 进行张量计算以兼容 Solver
        """
        # 1. 生成布朗运动增量 dW [batch, dim, N]
        dw_sample = torch.randn(num_sample, self.dim, self.num_time_interval, device=self.device) * self.sqrt_delta_t
        
        # 2. 初始化路径张量 X [batch, dim, N+1]
        x_sample = torch.zeros(num_sample, self.dim, self.num_time_interval + 1, device=self.device)
        x_sample[:, :, 0] = self.x_init.unsqueeze(0).expand(num_sample, self.dim)
        
        # 3. Euler-Maruyama 迭代 (论文 Eq.4 使用 Euler 格式 [cite: 87])
        # 注意: 这里是几何布朗运动 (GBM) 的离散化
        for i in range(self.num_time_interval):
            # X_{t+1} = X_t + mu * X_t * dt + sigma * X_t * dW
            x_curr = x_sample[:, :, i]
            dx = self.mu_bar * x_curr * self.delta_t + self.sigma * x_curr * dw_sample[:, :, i]
            x_sample[:, :, i + 1] = x_curr + dx
            
        return dw_sample, x_sample

    def f_torch(self, t, x, y, z):
        """
        生成元 Generator f (Driver Function)
        对应论文 Eq. [11] 中的非线性项 [cite: 137]
        f = -(1-delta)*Q(y)*y - R*y
        """
        # 计算 Q(y) - 分段线性函数 
        # Q(y) 在 (-inf, vh] 为 gammah
        # Q(y) 在 [vl, +inf) 为 gammal
        # 中间线性插值
        
        # 你的 Relu 技巧实现完全正确：
        # 当 y < vh: relu(neg) -> 0, output: gammah
        # 当 y > vl: inner part < gammal - gammah (neg), outer relu -> 0, output: gammal
        # 中间: 线性过渡
        piecewise_linear = torch.relu(
            torch.relu(y - self.vh) * self.slope + self.gammah - self.gammal
        ) + self.gammal

        # 论文 Eq [11]: -(1-delta)Q(u)u - Ru
        return (-(1 - self.delta) * piecewise_linear - self.rate) * y

    def g_torch(self, t, x):
        """
        终端条件 Terminal Condition
        对应论文: g(x) = min(x1, ..., xd) [cite: 138]
        """
        # x shape: [batch, dim] -> return [batch, 1]
        return torch.min(x, dim=1, keepdim=True).values
