import os
import json
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate_finetuned import evaluate_model

def scan_model_directories(base_dir):
    """扫描包含训练模型的所有目录，支持多层嵌套结构"""
    model_dirs = []
    
    # 检查基础目录是否存在
    if not os.path.exists(base_dir):
        print(f"错误: 基础目录 {base_dir} 不存在")
        return model_dirs
    
    # 递归查找所有checkpoint-*目录
    for root, dirs, files in os.walk(base_dir):
        # 检查当前目录是否为checkpoint目录
        if os.path.basename(root).startswith('checkpoint-'):
            # 检查是否存在相关模型文件
            has_model_files = any(f.endswith('.bin') for f in files)
            has_config = 'config.json' in files
            
            if has_model_files and has_config:
                print(f"找到模型检查点: {root}")
                model_dirs.append(root)
                continue
        
        # 检查当前目录是否包含pytorch_model.bin文件
        if 'pytorch_model.bin' in files and 'config.json' in files:
            print(f"找到模型: {root}")
            model_dirs.append(root)
    
    print(f"总共找到 {len(model_dirs)} 个训练好的模型")
    for model_dir in model_dirs:
        print(f"  - {model_dir}")
    
    return model_dirs

def evaluate_all_models(model_dirs, test_data_dir, output_dir):
    """评估所有模型并收集结果"""
    results = []
    
    for model_dir in model_dirs:
        print(f"评估模型: {model_dir}")
        try:
            # 运行评估
            result = evaluate_model(model_dir, test_data_dir, output_dir)
            
            # 添加模型目录信息
            result['model_dir'] = model_dir
            
            # 尝试寻找并加载results.json文件
            # 先在当前目录查找
            results_path = os.path.join(model_dir, 'results.json')
            
            # 如果不存在，则尝试在父目录查找（针对checkpoint目录）
            if not os.path.exists(results_path):
                parent_dir = os.path.dirname(model_dir)
                results_path = os.path.join(parent_dir, 'results.json')
            
            try:
                if os.path.exists(results_path):
                    with open(results_path, 'r') as f:
                        model_results = json.load(f)
                    
                    # 从results.json中提取config信息
                    if 'config' in model_results:
                        config = model_results['config']
                        result['batch_size'] = config.get('batch_size', 'N/A')
                        result['learning_rate'] = config.get('learning_rate', 'N/A')
                        result['frozen_layers'] = config.get('frozen_layers', 'N/A')
                        result['gradient_accumulation_steps'] = config.get('gradient_accumulation_steps', 'N/A')
                        
                        # 提取unfreeze_stages信息
                        if 'unfreeze_stages' in config:
                            stages = config['unfreeze_stages']
                            if isinstance(stages, list) and len(stages) > 0:
                                # 检查stages的结构
                                if isinstance(stages[0], dict) and 'num_layers' in stages[0]:
                                    result['unfrozen_layers'] = stages[0].get('num_layers', 'N/A')
                                elif isinstance(stages[0], list) and len(stages[0]) > 0 and isinstance(stages[0][0], dict):
                                    result['unfrozen_layers'] = stages[0][0].get('num_layers', 'N/A')
                                else:
                                    result['unfrozen_layers'] = 'N/A'
                            else:
                                result['unfrozen_layers'] = 'N/A'
                    else:
                        raise KeyError("results.json中没有config字段")
                else:
                    raise FileNotFoundError(f"未找到results.json文件: {results_path}")
                    
            except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                print(f"警告: 无法从 {results_path} 读取配置信息: {str(e)}")
                # 设置默认值
                result['batch_size'] = 'N/A'
                result['learning_rate'] = 'N/A'
                result['frozen_layers'] = 'N/A'
                result['gradient_accumulation_steps'] = 'N/A'
                result['unfrozen_layers'] = 'N/A'
            
            results.append(result)
        except Exception as e:
            print(f"评估模型 {model_dir} 时出错: {str(e)}")
    
    return results

def create_summary_report(results, output_dir):
    """创建模型评估摘要报告"""
    if not results:
        print("没有可用的评估结果")
        return
    
    # 创建DataFrame以便于排序和可视化
    df = pd.DataFrame(results)
    
    # 确保关键列存在
    for col in ['silhouette_score', 'model_dir', 'batch_size', 'learning_rate', 
               'frozen_layers', 'unfrozen_layers', 'train_loss', 'train_steps', 'train_time']:
        if col not in df.columns:
            print(f"警告: 结果中缺少 '{col}' 列，使用默认值")
            df[col] = 'N/A'
    
    # 按silhouette_score排序（降序）
    df_sorted = df.sort_values('silhouette_score', ascending=False)
    
    # 保存排序后的结果
    df_sorted.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # 找出最佳模型
    best_model = df_sorted.iloc[0]
    
    # 创建可序列化的字典，确保所有NumPy类型都被转换为标准Python类型
    best_model_dict = {
        'best_model_dir': str(best_model['model_dir']),
        'batch_size': str(best_model['batch_size']),
        'learning_rate': str(best_model['learning_rate']),
        'frozen_layers': str(best_model['frozen_layers']),
        'unfrozen_layers': str(best_model['unfrozen_layers'])
    }
    
    # 处理数值类型
    if best_model['silhouette_score'] != 'N/A':
        best_model_dict['silhouette_score'] = float(best_model['silhouette_score'])
    else:
        best_model_dict['silhouette_score'] = 'N/A'
        
    if best_model['train_loss'] != 'N/A':
        best_model_dict['train_loss'] = float(best_model['train_loss'])
    else:
        best_model_dict['train_loss'] = 'N/A'
        
    if best_model['train_steps'] != 'N/A':
        best_model_dict['train_steps'] = int(float(best_model['train_steps']))
    else:
        best_model_dict['train_steps'] = 'N/A'
        
    if best_model['train_time'] != 'N/A':
        best_model_dict['train_time'] = float(best_model['train_time'])
    else:
        best_model_dict['train_time'] = 'N/A'
    
    # 保存结果摘要文件
    with open(os.path.join(output_dir, 'best_model_summary.json'), 'w') as f:
        json.dump(best_model_dict, f, indent=4)
    
    # 创建可视化图表
    try:
        plt.figure(figsize=(12, 8))
        
        # 将模型路径简化为只显示run_X部分
        df_sorted['model_name'] = df_sorted['model_dir'].apply(lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))))
        
        # 绘制silhouette_score对比图
        sns.barplot(x='model_name', y='silhouette_score', data=df_sorted)
        plt.title('Model Performance Comparison (Silhouette Score)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    except Exception as e:
        print(f"警告: 创建可视化图表时出错: {str(e)}")
    
    print(f"\n评估摘要已保存到 {output_dir}")
    print(f"最佳模型: {best_model['model_dir']}")
    print(f"Silhouette Score: {best_model['silhouette_score']}")
    print(f"批次大小: {best_model['batch_size']}")
    print(f"学习率: {best_model['learning_rate']}")
    print(f"冻结层: {best_model['frozen_layers']}")
    print(f"解冻层: {best_model['unfrozen_layers']}")

def main():
    parser = argparse.ArgumentParser(description="比较多个训练模型的性能")
    parser.add_argument('--models_dir', type=str, required=True,
                      help='包含所有训练模型的基础目录')
    parser.add_argument('--test_data', type=str, required=True,
                      help='测试数据目录')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='保存评估结果的目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 手动指定模型目录 - 如果扫描失败，可以尝试这种方法
    if os.path.exists(args.models_dir) and len(scan_model_directories(args.models_dir)) == 0:
        print("尝试手动查找模型检查点...")
        model_dirs = []
        
        # 手动构建run_X/checkpoint-2176路径
        for i in range(5):  # 假设有5个run_X目录
            run_dir = os.path.join(args.models_dir, f"run_{i}")
            if os.path.exists(run_dir):
                checkpoint_dir = os.path.join(run_dir, "checkpoint-2176")
                if os.path.exists(checkpoint_dir):
                    print(f"手动找到模型: {checkpoint_dir}")
                    model_dirs.append(checkpoint_dir)
        
        if model_dirs:
            print(f"手动找到 {len(model_dirs)} 个模型")
        else:
            print("未能手动找到任何模型")
    else:
        # 原始扫描方法
        model_dirs = scan_model_directories(args.models_dir)
    
    # 评估所有模型
    results = evaluate_all_models(model_dirs, args.test_data, args.output_dir)
    
    # 创建摘要报告
    create_summary_report(results, args.output_dir)

if __name__ == "__main__":
    main() 