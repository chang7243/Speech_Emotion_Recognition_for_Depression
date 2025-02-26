import json
import itertools
import os
from pathlib import Path

def generate_hyperparameter_configs(output_path='configs/all_configs.json'):
    """生成超参数配置并保存到JSON文件，适配冻结层继续预训练"""
    # 定义核心超参数搜索空间
    hyperparameters = {
        # 基础训练参数
        "batch_size": [4, 8],
        "learning_rate": [1e-5, 5e-6, 1e-6],
        
        # 优化器参数
        "weight_decay": [0.0001, 0.001],
        
        # 模型参数
        "hidden_dropout": [0.1, 0.2],
        
        # 训练策略
        "gradient_accumulation_steps": [2, 4],
        
        
        # 冻结层参数 - 指定要冻结的底部层数
        "frozen_layers": [6, 8, 10, 12],
        
        # 梯度检查点
        "gradient_checkpointing": [True],
        
        # 渐进式解冻相关参数 - 简化为单阶段
        "unfreeze_stages": [
            # 渐进式方案: 逐步解冻更多层，总共5个epoch
            [
                {'epochs': 2, 'num_layers': 2, 'lr_multiplier': 1.0},
                {'epochs': 3, 'num_layers': 4, 'lr_multiplier': 0.8}
            ],
            # 另一个渐进式方案选项
            [
                {'epochs': 1, 'num_layers': 2, 'lr_multiplier': 1.0},
                {'epochs': 2, 'num_layers': 3, 'lr_multiplier': 0.5},
                {'epochs': 2, 'num_layers': 4, 'lr_multiplier': 0.2}
            ]
        ],
        
        # 早停参数
        "early_stopping_patience": [2, 3]
    }

    # 基础配置
    base_config = {
        "data_dir": "/kaggle/working/train",
        "use_kaggle": True,
        "log_interval": 10,
        "save_model": True,
        "fp16": False,  # 使用混合精度训练
    }

    # 生成所有可能的超参数组合
    configs = []
    run_id = 0

    # 生成主要参数组合
    main_params = list(itertools.product(
        hyperparameters["batch_size"],
        hyperparameters["learning_rate"],
        hyperparameters["weight_decay"],
        hyperparameters["hidden_dropout"],
        hyperparameters["gradient_accumulation_steps"],
        hyperparameters["frozen_layers"],
        hyperparameters["gradient_checkpointing"],
        hyperparameters["unfreeze_stages"],
        hyperparameters["early_stopping_patience"]
    ))

    # 筛选有效的配置
    for (batch_size, lr, weight_decay, hidden_dropout, 
         grad_accum_steps, frozen_layers, 
         gradient_checkpointing, unfreeze_stages, patience) in main_params:
        
        # 确保unfreeze_stages中的num_layers与frozen_layers兼容
        max_unfrozen_layers = 12 - frozen_layers  # 假设共12层
        
        # 如果unfreeze_stages中的num_layers大于可解冻的层数，则跳过
        if any(stage['num_layers'] > max_unfrozen_layers for stage in unfreeze_stages):
            continue
            
        # 计算总训练轮数
        if isinstance(unfreeze_stages, list) and len(unfreeze_stages) > 0:
            stage_epochs = sum(stage['epochs'] for stage in unfreeze_stages)
            total_epochs = max(stage_epochs, 5)  # 确保至少5个epoch
        else:
            total_epochs = 5
        
        config = base_config.copy()
        config.update({
            "batch_size": batch_size,
            "learning_rate": lr,
            "total_epochs": total_epochs,
            "output_dir": f"/kaggle/working/models/run_{run_id}",
            "log_file": f"fine_tuning_run_{run_id}.log",
            "weight_decay": weight_decay,
            "hidden_dropout": hidden_dropout,
            "gradient_accumulation_steps": grad_accum_steps,
            "frozen_layers": frozen_layers,
            "gradient_checkpointing": gradient_checkpointing,
            "unfreeze_stages": unfreeze_stages,
            "early_stopping_patience": patience
        })
        
        configs.append(config)
        run_id += 1

    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存配置到文件
    with open(output_path, 'w') as f:
        json.dump(configs, f, indent=4)

    print(f"生成了 {len(configs)} 种不同的配置")
    print(f"配置文件已保存到 {output_path}")
    
    return configs

def filter_recommended_configs(configs, max_configs=10):
    """筛选推荐的配置组合，减少总数量"""
    # 根据经验选择更可能有效的配置
    filtered_configs = []
    
    # 优先选择较小的batch_size和frozen_layers=8的配置
    for config in configs:
        if (config["batch_size"] <= 16 and 
            config["frozen_layers"] == 8 and
            config["learning_rate"] >= 5e-5):
            filtered_configs.append(config)
            if len(filtered_configs) >= max_configs:
                break
    
    # 如果还没有足够的配置，添加更多
    if len(filtered_configs) < max_configs:
        for config in configs:
            if config not in filtered_configs:
                filtered_configs.append(config)
                if len(filtered_configs) >= max_configs:
                    break
    
    return filtered_configs

def load_configs(config_path):
    """加载超参数配置"""
    with open(config_path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成超参数配置")
    parser.add_argument('--output', type=str, default='configs/all_configs.json',
                      help='保存配置的路径')
    parser.add_argument('--filter', action='store_true',
                      help='是否筛选推荐的配置')
    parser.add_argument('--max_configs', type=int, default=10,
                      help='筛选后保留的最大配置数量')
    
    args = parser.parse_args()
    
    # 生成配置
    configs = generate_hyperparameter_configs(args.output)
    
    # 如果需要，筛选配置
    if args.filter:
        filtered_configs = filter_recommended_configs(configs, args.max_configs)
        filtered_output = args.output.replace('.json', '_filtered.json')
        
        with open(filtered_output, 'w') as f:
            json.dump(filtered_configs, f, indent=4)
            
        print(f"筛选后保留了 {len(filtered_configs)} 种配置")
        print(f"筛选后的配置已保存到 {filtered_output}")