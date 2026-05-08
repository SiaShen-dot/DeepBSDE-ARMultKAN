import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import yaml  # 需要 PyYAML 库

from config import Config
from equation import PricingDefaultRisk, HJBLQ 
from model import BSDE_ARMultKAN_Model

def load_config_from_yaml(model_path, eval_batch_size=2048):
    """
    从模型路径同级目录加载 config.yml，并转换为 namespace 对象
    """
    # 1. 定位 config.yml
    experiment_dir = os.path.dirname(model_path)
    config_path = os.path.join(experiment_dir, 'config.yml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    print(f"Loading config from: {config_path}")
    
    # 2. 读取 YAML
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    # 3. 覆盖/修正部分参数以适应评估环境
    # (评估时通常可以使用更大的 batch_size，且不需要训练相关的参数)
    config_dict['batch_size'] = eval_batch_size
    
    # 4. 类型转换 (String -> Torch Object)
    # 训练保存的是字符串 'float32'，但在代码中我们需要 torch.float32
    if config_dict.get('dtype') == 'float64':
        config_dict['dtype'] = torch.float64
    else:
        config_dict['dtype'] = torch.float32 # 默认
        
    # 强制重新检测 Device (防止训练时用了 cuda:0 而评估时想用 cpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_dict['device'] = device
    
    # 5. 转换为 argparse.Namespace 对象
    # 因为你的 Config 类 (cfg = Config(args)) 通常期望输入是一个对象而不是字典
    args = argparse.Namespace(**config_dict)
    
    # 补全可能在训练代码中动态生成但未保存到 yaml 的属性
    # 例如 train.py 中有一行: args.width = [args.dim] + args.hidden_layers + [args.dim]
    # 如果 yaml 里没有 width，需要这里补上
    if not hasattr(args, 'width') or args.width is None:
        args.width = [args.dim] + args.hidden_layers + [args.dim]
        
    return args

def evaluate_model():
    # 只需要接收一个参数：模型路径
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model.pth')
    # 可选：如果你想临时改变评估时的 batch size
    parser.add_argument('--eval_batch_size', type=int, default=2048) 
    
    cmd_args = parser.parse_args()

    experiment_dir = os.path.dirname(cmd_args.model_path)
    fig_path = os.path.join(experiment_dir, 'eval_result.png')
    
    # --- 核心修改：从 YAML 加载配置 ---
    args = load_config_from_yaml(cmd_args.model_path, cmd_args.eval_batch_size)
    device = args.device
    
    print(f"Model Configuration Loaded: {args.eqn_name} (Dim={args.dim})")

    # 1. 初始化 Config 对象
    cfg = Config(args)
    
    # 2. 初始化方程
    if args.eqn_name == 'PricingDefaultRisk':
        bsde = PricingDefaultRisk(cfg)
    elif args.eqn_name == 'HJBLQ':
        bsde = HJBLQ(cfg)
    else:
        raise ValueError(f"Unknown equation: {args.eqn_name}")
        
    # 3. 初始化模型结构
    model = BSDE_ARMultKAN_Model(cfg, bsde).to(device)

    # 4. 加载权重
    print(f"Loading weights from {cmd_args.model_path}...")
    checkpoint = torch.load(cmd_args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # ---------------------------------------------------
    # 下面是评估逻辑 (与之前相同，直接粘贴即可)
    # ---------------------------------------------------
    with torch.no_grad():
        dw, x = bsde.sample(args.batch_size)
        inputs = (dw, x)
        y_terminal_pred = model(inputs)
        y_terminal_true = bsde.g_torch(bsde.total_time, x[:, :, -1])

    y_pred_np = y_terminal_pred.cpu().numpy().flatten()
    y_true_np = y_terminal_true.cpu().numpy().flatten()
    
    print("\n" + "="*40)
    print("      STATISTICAL DIAGNOSIS      ")
    print("="*40)
    print(f"{'Metric':<10} | {'Prediction':<18} | {'Ground Truth':<18}")
    print("-" * 52)
    print(f"{'Mean':<10} | {np.mean(y_pred_np):.4f}{'':<12} | {np.mean(y_true_np):.4f}")
    print(f"{'Std':<10} | {np.std(y_pred_np):.4f}{'':<12} | {np.std(y_true_np):.4f}")
    print("-" * 52)
    print(f"Y0 (Theta): {model.y_init.item():.4f}")
    
    # 画图部分

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_true_np, y_pred_np, alpha=0.5, s=2)
    lims = [np.min([plt.xlim(), plt.ylim()]), np.max([plt.xlim(), plt.ylim()])]
    plt.plot(lims, lims, 'r-', alpha=0.75)
    plt.title(f"{args.eqn_name}: Pred vs True")
    plt.xlabel("True Terminal Value")
    plt.ylabel("Model Predicted")
    
    plt.subplot(1, 2, 2)
    plt.hist(y_true_np, bins=50, alpha=0.5, label='True', density=True)
    plt.hist(y_pred_np, bins=50, alpha=0.5, label='Pred', density=True)
    plt.legend()
    plt.title("Distribution")
    
    plt.tight_layout()
    
    plt.savefig(fig_path)
    print("Saved eval_result.png")

if __name__ == "__main__":
    evaluate_model()