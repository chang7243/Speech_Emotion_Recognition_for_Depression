import torch
from transformers import Wav2Vec2ForPreTraining, Wav2Vec2FeatureExtractor
import numpy as np
import os
import json
import logging
from pathlib import Path
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torchaudio
from sklearn.manifold import TSNE

def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

def load_model_and_processor(model_path):
    """加载模型和特征提取器"""
    model = Wav2Vec2ForPreTraining.from_pretrained(model_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')
    return model, feature_extractor

def extract_features(model, processor, audio_file, device):
    """提取音频特征"""
    waveform, sample_rate = torchaudio.load(audio_file)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)
    else:
        waveform = waveform.squeeze(0)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    inputs = processor(
        waveform, 
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    
    if 'input_values' in inputs:
        inputs = inputs.to(device)
    else:
        inputs = {'input_values': inputs.to(device)}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
        if hasattr(outputs, 'projected_states'):
            features = outputs.projected_states.mean(dim=1).cpu().numpy()
        elif hasattr(outputs, 'extract_features'):
            features = outputs.extract_features.mean(dim=1).cpu().numpy()
        else:
            features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
        return features

def setup_matplotlib_fonts():
    """设置matplotlib支持中文"""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    
    # 尝试使用不同的中文字体
    try:
        # 在Windows上尝试微软雅黑
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except:
        # 如果找不到字体，使用简单英文标题
        logging.warning("找不到中文字体，将使用英文标题")

def evaluate_model(model_dir, test_data_dir, output_dir):
    """评估微调后的模型"""
    # 设置matplotlib字体
    setup_matplotlib_fonts()
    
    # 首先尝试从trainer_state.json加载基本训练状态
    try:
        with open(os.path.join(model_dir, 'trainer_state.json'), 'r') as f:
            train_state = json.load(f)
            
            # 尝试从不同位置提取训练损失
            if 'train_loss' not in train_state:
                # 从log_history中提取最后一个epoch的损失
                if 'log_history' in train_state and len(train_state['log_history']) > 0:
                    train_state['train_loss'] = train_state['log_history'][-1].get('loss', 0.0)
                # 从best_metric中提取
                elif 'best_metric' in train_state:
                    train_state['train_loss'] = train_state.get('best_metric', 0.0)
                else:
                    train_state['train_loss'] = 0.0
                    
            # 确保有训练步数
            if 'train_steps' not in train_state:
                train_state['train_steps'] = train_state.get('global_step', 0)
                
            # 确保有训练时间
            if 'train_time' not in train_state:
                train_state['train_time'] = train_state.get('total_train_time', 0.0)
    except FileNotFoundError:
        logging.warning(f"未找到trainer_state.json文件，将创建默认状态")
        train_state = {
            "train_loss": 0.0,
            "train_steps": 0,
            "train_time": 0.0
        }
    
    # 确定results.json的位置
    # 先检查模型目录是否是checkpoint目录
    model_dir_name = os.path.basename(model_dir)
    if model_dir_name.startswith('checkpoint-'):
        # 如果是检查点目录，则results.json在父目录中
        results_path = os.path.join(os.path.dirname(model_dir), 'results.json')
    else:
        # 否则先尝试同级目录
        results_path = os.path.join(model_dir, 'results.json')
        if not os.path.exists(results_path):
            # 如果同级目录不存在，尝试父目录
            results_path = os.path.join(os.path.dirname(model_dir), 'results.json')
    
    # 尝试加载results.json
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
            config = results.get('config', {})
            logging.info(f"从 {results_path} 加载配置")
    except FileNotFoundError:
        logging.warning(f"未找到results.json文件 ({results_path})，将使用空配置")
        config = {}
    
    # 将配置添加到train_state
    train_state['config'] = config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    model, processor = load_model_and_processor(model_dir)
    model = model.to(device)
    model.eval()
    
    features = []
    audio_files = [f for f in os.listdir(test_data_dir) if f.endswith('.wav')]
    
    for audio_file in tqdm(audio_files, desc="提取特征中"):
        feature = extract_features(
            model, 
            processor, 
            os.path.join(test_data_dir, audio_file),
            device
        )
        features.append(feature)
    
    # 将特征转换为numpy数组
    features = np.vstack(features)
    
    # 计算特征的相似度（使用轮廓系数）
    if len(features) > 2:  # 需要至少3个样本计算轮廓系数
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(5, len(features)), random_state=0).fit(features)
        silhouette_avg = silhouette_score(features, kmeans.labels_)
    else:
        silhouette_avg = 0
        logging.warning("样本数量不足，无法计算轮廓系数")
    
    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # 绘制 t-SNE 图
    plt.figure(figsize=(10, 8))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=kmeans.labels_, cmap='viridis', s=50)
    plt.title('t-SNE Visualization of Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(label='Cluster Label')
    plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))
    plt.close()
    
    # 保存特征
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'features.npy'), features)
    
    # 可视化特征
    plt.figure(figsize=(16, 6))
    
    # 特征相关性热图
    plt.subplot(1, 2, 1)
    correlation_matrix = np.corrcoef(features.T)
    sns.heatmap(correlation_matrix[:10, :10], cmap='coolwarm')
    plt.title("Feature Correlation (First 10x10)")
    
    # 特征分布图
    plt.subplot(1, 2, 2)
    feature_means = features.mean(axis=0)[:10]
    plt.bar(range(10), feature_means)
    plt.title("Feature Distribution (First 10)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_wav2vec2.png'))
    plt.close()
    
    # 保存评估结果
    evaluation_result = {
        'silhouette_score': float(silhouette_avg),
        'feature_dim': features.shape[1],
        'num_samples': len(features),
        'train_loss': train_state.get('train_loss', 0.0),
        'train_steps': train_state.get('train_steps', 0),
        'train_time': train_state.get('train_time', 0.0),
        'config': config
    }
    
    with open(os.path.join(output_dir, 'evaluation_wav2vec2.json'), 'w') as f:
        json.dump(evaluation_result, f, indent=4)
    
    return evaluation_result

def main():
    import argparse
    parser = argparse.ArgumentParser(description="评估微调后的模型")
    parser.add_argument('--model_dir', type=str, required=True,
                      help='包含微调模型的目录')
    parser.add_argument('--test_data', type=str, required=True,
                      help='测试数据目录')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='保存评估结果的目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    setup_logging(os.path.join(args.output_dir, 'evaluation.log'))
    
    # 评估模型
    result = evaluate_model(args.model_dir, args.test_data, args.output_dir)
    
    # 打印评估结果
    print("\n评估结果:")
    print(f"轮廓系数: {result['silhouette_score']:.4f}")
    print(f"训练损失: {result['train_loss']:.4f}")
    print(f"训练步数: {result['train_steps']}")
    print(f"训练时间: {result['train_time']:.2f}秒")

if __name__ == "__main__":
    main() 