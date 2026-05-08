import os
import json
import yaml
import argparse
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import time
import csv

# 导入之前的模块
from config import Config
from equation import *
from model import BSDE_ARMultKAN_Model
from solver import BSDESolver

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def str2list(v):
    """辅助函数：处理命令行传入的列表参数，支持 [1,2] 或 1 2 格式"""
    if isinstance(v, list):
        return v
    if v.startswith('['):
        return json.loads(v)
    return [int(x) for x in v.split(',')]

def get_args():
    parser = argparse.ArgumentParser(description="Deep BSDE Solver with ARMultKAN")

    # --- 实验管理 ---
    parser.add_argument('--name', type=str, default='HJBLQ_100D', help='Experiment name')
    parser.add_argument('--model_dir', type=str, default='./models', help='Root directory for saving models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--load_ckpt', type=str, default=None, help='Path to checkpoint to resume/load')
    parser.add_argument('--load_optim', action='store_true', help='Load optimizer and scheduler')

    # --- 方程配置 (参考 hjb_lq_d100.json) ---
    parser.add_argument('--eqn_name', type=str, default='HJBLQ', choices=['HJBLQ', 'PricingDefaultRisk'], help='Equation name')
    parser.add_argument('--dim', type=int, default=100, help='Dimension')
    parser.add_argument('--total_time', type=float, default=1.0, help='Total time horizon T')
    parser.add_argument('--num_time_interval', type=int, default=20, help='Number of time intervals N')

    # --- 网络配置 (参考 net_config) ---
    parser.add_argument('--y_init_range', type=str2list, default=[0, 1], help='Range for initial Y guess')
    # 原始 JSON 是 num_hiddens: [110, 110]，这里指中间层
    parser.add_argument('--hidden_layers', type=str2list, default=[110, 110], help='Width of hidden layers')
    parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='Minimum learning rate for scheduler')
    parser.add_argument('--num_iterations', type=int, default=2000, help='Total training iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--valid_size', type=int, default=256, help='Validation set size')
    parser.add_argument('--logging_frequency', type=int, default=1, help='Logging frequency')
    parser.add_argument('--dtype', type=str, default='float64', choices=['float32', 'float64'], help='Tensor float precision')
    parser.add_argument('--verbose', action='store_true', default=True, help='Print logs')

    # --- RMultKAN 特定参数 ---
    parser.add_argument('--grid', type=int, default=5, help='Initial grid size')
    parser.add_argument('--k', type=int, default=3, help='Spline order')
    parser.add_argument('--eps', type=float, default=0.02, help='Initial grid eps')

    # --- AGU (自适应网格) 参数 ---
    parser.add_argument('--use_agu', action='store_true', help='Enable Adaptive Grid Update')
    parser.add_argument('--agu_freq', type=int, default=500, help='Frequency of AGU updates')
    parser.add_argument('--agu_warmup', type=int, default=500, help='Steps before starting AGU')
    parser.add_argument('--agu_ratio', type=float, default=0.2, help='Ratio of adaptive samples vs uniform samples')
    parser.add_argument('--loss_window_size', type=int, default=10, help='Window size of loss history')

    return parser.parse_args()

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # 1. 获取参数
    args = get_args()

    # 自动拼接：[100] + [110, 110] + [100]
    args.width = [args.dim] + args.hidden_layers + [args.dim]
    
    # 2. 设置随机种子
    set_seed(args.seed)
    
    # 3. 创建实验目录
    experiment_dir = os.path.join(args.model_dir, args.name)
    os.makedirs(experiment_dir, exist_ok=True)

    # 定义路径
    best_ckpt_path = os.path.join(experiment_dir, 'model.pth')
    log_file = os.path.join(experiment_dir, 'log.csv')
    config_path = os.path.join(experiment_dir, 'config.yml')
    
    # 4. 保存 Config
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False, sort_keys=False)
    print(f"Configuration saved to {config_path}")

    print('-' * 20)
    for key in vars(args):
        print('%s: %s' % (key, vars(args)[key]))
    print('-' * 20)

    # 5. 初始化配置对象 (将 args 转化为内部 Config 类)
    cfg = Config(args)

    # 6. 初始化方程
    # 根据名称动态选择方程类
    if args.eqn_name == 'HJBLQ':
        bsde = HJBLQ(cfg)
    elif args.eqn_name == 'PricingDefaultRisk':
        bsde = PricingDefaultRisk(cfg) 
    else:
        raise NotImplementedError(f"Equation {args.eqn_name} not implemented")

    # 7. 初始化模型与求解器
    model = BSDE_ARMultKAN_Model(cfg, bsde).to(device)
    solver = BSDESolver(cfg, bsde, model)
    
    # 优化器与学习率调度
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 推荐的 CosineAnnealingLR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_iterations,  # T_max 必须设置为总迭代步数
        eta_min=args.eta_min        # 设置一个最小学习率，防止后期完全停止更新
    )
    
    start_time = time.time()
    
    # 生成固定的验证集
    valid_data = bsde.sample(args.valid_size)

    print(f"Model: RMultKAN with AGU={args.use_agu}")
    
    # 9. 断点续训逻辑 (Resume)
    start_step = 0
    best_l2 = float('inf')

    if args.load_ckpt and os.path.exists(args.load_ckpt):
        print(f"Loading checkpoint: {args.load_ckpt}")
        
        ckpt = torch.load(args.load_ckpt, map_location=device)
        
        # 加载模型权重
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        
        # 加载优化器状态 (仅当不是微调时)
        if args.load_optim:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        
        start_step = ckpt.get('epoch', 0) + 1
        best_l2 = ckpt.get('l2', float('inf'))
        print(f"Resumed from step {start_step}, Best L2: {best_l2:.4%}")
    
    # 10. 准备日志文件
    # 如果是续训，使用追加模式 'a'，否则覆盖 'w'
    if start_step > 0 and os.path.exists(log_file):
        log_file_handle = open(log_file, 'a', newline='')
        csv_writer = csv.writer(log_file_handle)
    else:
        log_file_handle = open(log_file, 'w', newline='')
        csv_writer = csv.writer(log_file_handle)
        csv_writer.writerow(['Step', 'Loss', 'Y0_Pred', 'Rel_Error', 'Time'])

    # 生成固定的验证数据
    valid_dw, valid_x = bsde.sample(args.valid_size)
    valid_data = (valid_dw, valid_x)
    
    start_time = time.time()
    print(f"Start Training {bsde.__class__.__name__}...")

    # --- 训练循环 ---
    pbar = tqdm(total=args.num_iterations, initial=start_step, desc='Training', ncols=120)
    
    rel_error = 1.0
    history = []
    for step in range(start_step, args.num_iterations + 1):
        
        # A. 采样
        train_dw, train_x = bsde.sample(args.batch_size)
        train_data = (train_dw, train_x)

        # B. AGU 更新
        # 确保 history 变量在 loop 外部初始化了，例如: history_losses = [] 
        # (通常你可以用 valid_loss 或者 train_loss 的滑动平均)
        
        # 在 loop 内部收集 loss (如果没有现成的 list)
        # 注意：这里建议传入 validation loss 或者 training loss 的平滑值
        # 假设你有一个 train_loss_history 列表记录了每个 step 的 loss
        
        if args.use_agu and step >= args.agu_warmup:
            if step % args.agu_freq == 0:
                # 传入 history 以便计算动态 eps
                # 传入 agu_ratio (需要在 get_args 添加参数，或者直接写死)
                model.update_grids(
                    train_x, 
                    loss_history=history, # 这里的 history 应该是你记录 loss 值的列表
                    agu_ratio=args.agu_ratio,
                    window_size=args.loss_window_size
                )
                
                if args.verbose:
                    # 可以打印一下当前的 eps 看看变化
                    current_eps = model.subnets[0].grid_eps
                    print(f"\n --- [Step {step}] AGU Triggered | Eps: {current_eps:.4f} ---")

        # C. 优化步
        optimizer.zero_grad()
        # 使用 solver 中定义的 loss_fn
        loss = solver.loss_fn(train_data)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # --- 新增：记录 Loss 用于 AGU ---
        history.append(loss.item())
        if len(history) > 2 * args.loss_window_size + 1: # 保持列表不要无限增长，保留最近的即可
            history.pop(0)

        # D. 验证、日志与保存
        if step % args.logging_frequency == 0 or step == args.num_iterations:
            with torch.no_grad():
                valid_loss = solver.loss_fn(valid_data).item()
                y0_pred = model.y_init.item()
                elapsed = time.time() - start_time
                
                # 相对误差计算
                if bsde.y_init is not None:
                    ref = bsde.y_init if not isinstance(bsde.y_init, torch.Tensor) else bsde.y_init.item()
                    rel_error = abs(y0_pred - ref) / abs(ref)

                # 写入 CSV
                csv_writer.writerow([step, valid_loss, y0_pred, rel_error, elapsed])
                log_file_handle.flush() # 强制刷新缓冲区，防止程序崩溃丢失日志

                # 保存最佳模型
                if rel_error < best_l2 and step > 0:
                    best_l2 = rel_error
                    torch.save({
                        'epoch': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'l2': best_l2,
                    }, best_ckpt_path)
                    #if args.verbose:
                        #print(f"\n => Saved Best Model. Rel Error: {best_l2:.2%}")

        # 临时插入 train.py 用于调试
        if step % 100 == 0:
            with torch.no_grad():
                # 1. 拿一个 batch 的数据
                inputs = bsde.sample(100) 
                dw, x = inputs
                
                # 2. 预测终端值
                y_pred = model(inputs) # Shape [100, 1]
                
                # 3. 真实终端值
                y_true = bsde.g_torch(bsde.total_time, x[:, :, -1]) # Shape [100, 1]
                
                print("\n--- Diagnostic Check ---")
                print(f"Y_Pred Mean: {y_pred.mean().item():.2f} | Std: {y_pred.std().item():.2f}")
                print(f"Y_True Mean: {y_true.mean().item():.2f} | Std: {y_true.std().item():.2f}")
                print(f"Max Diff: {(y_pred - y_true).abs().max().item():.2f}")

        pbar.set_description(f"Step: [{step}/{args.num_iterations}] Loss: {valid_loss:.4e} | Err: {rel_error:.2%}")
        pbar.update(1)

    pbar.close()
    log_file_handle.close()
    print(f"Training Finished/Stopped. Logs saved to {log_file}")
        
    # 打印最终结果
    if bsde.y_init is not None:
        ref = bsde.y_init if not isinstance(bsde.y_init, torch.Tensor) else bsde.y_init.item()
        print(f"Final Y0: {model.y_init.item():.4f} | Ref: {ref:.4f} | Best Err: {best_l2:.6f}")

if __name__ == "__main__":
    main()