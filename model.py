import numpy as np
import torch
import torch.nn as nn
import math

# 尝试导入 RMultKAN，如果不存在则报错提示
try:
    from rkan import RMultKAN
except ImportError:
    raise ImportError("请确保目录中存在 rkan 文件夹且包含 RMultKAN 类。")

class BSDE_ARMultKAN_Model(nn.Module):
    def __init__(self, config, bsde):
        super().__init__()
        self.config = config
        self.bsde = bsde
        
        # --- 待优化的全局变量 ---
        # 1. 初始时刻的解 Y_0 (标量)
        low, high = config.y_init_range
        self.y_init = nn.Parameter(torch.tensor(np.random.uniform(low, high), dtype=config.dtype))
        
        # 2. 初始时刻的梯度 Z_0 (向量, 1 x dim)
        self.z_init = nn.Parameter(torch.zeros(1, config.dim, dtype=config.dtype).uniform_(-0.1, 0.1))
        
        # --- 时间堆叠的子网络 ---
        # 对应每个时间步 t=0 到 t=N-2 的梯度逼近器
        self.subnets = nn.ModuleList()
        for _ in range(bsde.num_time_interval - 1):
            arkan = RMultKAN(
                width=config.width, 
                grid=config.grid, 
                k=config.k, 
                seed=config.seed,
                grid_eps=config.eps,
                device=config.device
            )
            self.subnets.append(arkan)

    def forward(self, inputs):
        """
        前向传播：根据 SDE 路径计算终端时刻的预测解 Y_T
        inputs: (dw, x)
        """
        dw, x = inputs
        batch_size = dw.shape[0]
        
        # 初始化
        y = torch.ones(batch_size, 1, device=self.config.device) * self.y_init
        z = torch.ones(batch_size, 1, device=self.config.device) @ self.z_init 
        
        # 时间步循环
        for t in range(0, self.bsde.num_time_interval - 1):
            # 1. 更新 Y (根据 BSDE 离散格式)
            f_val = self.bsde.f_torch(t * self.bsde.delta_t, x[:, :, t], y, z)
            z_dw = torch.sum(z * dw[:, :, t], dim=1, keepdim=True)
            y = y - self.bsde.delta_t * f_val + z_dw
            
            # 2. 更新 Z (使用当前时间步的 ARKAN 子网络预测)
            # 注意：子网络输入是下一时刻的状态 X_{t+1}
            if self.config.eqn_name == 'PricingDefaultRisk':
                z = self.subnets[t](x[:, :, t + 1]) / math.sqrt(self.config.dim)
            else:
                z = self.subnets[t](x[:, :, t + 1]) / self.config.dim 
            
        # 最后一个时间步的处理 (不需要预测下一个 Z)
        t = self.bsde.num_time_interval - 1
        f_val = self.bsde.f_torch(t * self.bsde.delta_t, x[:, :, t], y, z)
        z_dw = torch.sum(z * dw[:, :, t], dim=1, keepdim=True)
        y_terminal = y - self.bsde.delta_t * f_val + z_dw
        
        return y_terminal

    def update_grids(self, x_path, loss_history=None, agu_ratio=0.2, window_size=10):
        """
        改进版 AGU (Adaptive Grid Update)
        结合了 PINN 的动态 EPS 调整和混合采样策略。
        
        Args:
            x_path: [batch, dim, N+1] 粒子路径
            loss_history: list, 历史 loss 记录 (用于动态调整 eps)
            agu_ratio: float, 混合采样中均匀分布样本的比例 (默认 0.2)
            window_size: int, 判定 loss 变化的窗口大小
        """
        # -----------------------------
        # Part 1: 动态调整 Grid EPS
        # -----------------------------
        # 逻辑：Loss 震荡/下降快 -> 增大 eps (保持平滑/探索)
        #       Loss 趋于稳定 -> 减小 eps (精细化/开发)
        if loss_history is not None and len(loss_history) > 2 * window_size:
            # 计算近期和远期的平均 Loss
            recent_loss = sum(loss_history[-window_size:]) / window_size
            prev_loss = sum(loss_history[-2*window_size : -window_size]) / window_size
            
            loss_diff = prev_loss - recent_loss
            
            # 判断调整方向
            # 如果 diff 很小 (< 1e-3)，说明收敛停滞 -> 减小 eps，让网格更贴近数据
            # 如果 diff 很大或为负 (震荡) -> 增大 eps，让网格更平滑
            # 注意：这里我们假设所有 subnets 共享相似的 eps 策略，或者我们可以分别为它们调整
            
            # 获取第一个子网络的 eps 作为参考 (假设大家初始一样)
            current_eps = self.subnets[0].grid_eps
            
            if abs(loss_diff) < 1e-3:
                new_eps = max(0.001, current_eps - 0.005) # 下限设为 0.01 防止除零或过拟合
            else:
                new_eps = min(1.0, current_eps + 0.005)
            
            # 将新的 eps 应用到所有时间步的子网络
            for subnet in self.subnets:
                subnet.grid_eps = new_eps

        # -----------------------------
        # Part 2: 混合采样更新网格
        # -----------------------------
        with torch.no_grad():
            for t in range(len(self.subnets)):
                # A. 获取真实粒子分布 (Adaptive part)
                # x shape: [batch, dim]
                real_samples = x_path[:, :, t + 1]
                batch_size, dim = real_samples.shape
                
                # B. 生成均匀分布样本 (Uniform part) - 防止边界坍缩
                # 我们根据当前 batch 的 min/max 确定包围盒，并在其中均匀采样
                # 样本数量由 agu_ratio 决定
                n_uniform = int(batch_size * agu_ratio)
                
                if n_uniform > 0:
                    # 计算包围盒 (加一点 margin 0.1 确保覆盖边界)
                    batch_min = real_samples.min(dim=0)[0] - 0.1
                    batch_max = real_samples.max(dim=0)[0] + 0.1
                    
                    # 生成均匀噪声: [n_uniform, dim]
                    uniform_samples = torch.rand(n_uniform, dim, device=self.config.device)
                    uniform_samples = uniform_samples * (batch_max - batch_min) + batch_min
                    
                    # C. 混合数据
                    combined_samples = torch.cat([real_samples, uniform_samples], dim=0)
                else:
                    combined_samples = real_samples

                # D. 调用 KAN 的 update_grid
                self.subnets[t].update_grid(combined_samples)