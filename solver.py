import torch

class BSDESolver(object):
    def __init__(self, config, bsde, model):
        self.config = config
        self.bsde = bsde
        self.model = model

    def loss_fn(self, inputs):
        """计算 Deep BSDE 的损失"""
        dw, x = inputs
        # 1. 模型预测的终端值
        y_terminal_pred = self.model(inputs)
        # 2. 真实的终端条件 g(X_T)
        y_terminal_true = self.bsde.g_torch(self.bsde.total_time, x[:, :, -1])
        
        delta = y_terminal_pred - y_terminal_true
        
        # Clipped MSE Loss (防止梯度爆炸)
        DELTA_CLIP = 50.0
        loss = torch.mean(torch.where(
            torch.abs(delta) < DELTA_CLIP,
            torch.square(delta),
            2 * DELTA_CLIP * torch.abs(delta) - DELTA_CLIP ** 2
        ))
        return loss