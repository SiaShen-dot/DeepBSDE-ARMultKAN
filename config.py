# 文件名: config.py
import torch
import json

class Config:
    def __init__(self, args):
        """
        根据 argparse 的 args 初始化配置
        """
        # --- 基础参数拷贝 ---
        self.eqn_name = args.eqn_name
        self.dim = args.dim
        self.total_time = args.total_time
        self.num_time_interval = args.num_time_interval
        
        self.num_iterations = args.num_iterations
        self.batch_size = args.batch_size
        self.valid_size = args.valid_size
        self.logging_frequency = args.logging_frequency
        self.verbose = args.verbose
        self.y_init_range = args.y_init_range

        # --- 优化器参数 ---
        self.lr = args.lr
        self.lr_gamma = 0.1       

        # --- 设备与精度处理 ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if args.dtype == 'float64':
            self.dtype = torch.float64
            torch.set_default_dtype(torch.float64)
        else:
            self.dtype = torch.float32
            torch.set_default_dtype(torch.float32)

        # --- KAN 网络结构构建 ---
        if hasattr(args, 'width'):
             self.width = args.width
        else:
             # 兜底逻辑
             self.width = [self.dim] + args.hidden_layers + [self.dim]
        
        self.grid = args.grid
        self.k = args.k
        self.eps = args.eps
        self.seed = getattr(args, 'seed', 42)
        
        # --- AGU 配置 ---
        self.use_agu = args.use_agu
        self.agu_freq = args.agu_freq
        self.agu_warmup = args.agu_warmup
        self.agu_ratio = args.agu_ratio
        self.loss_window_size = args.loss_window_size