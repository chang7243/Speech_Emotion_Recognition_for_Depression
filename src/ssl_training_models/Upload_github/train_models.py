import torch
from transformers import (
    Wav2Vec2ForPreTraining,
    Trainer, 
    TrainingArguments,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Config
)
from datasets import load_dataset
import os
import json
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
from transformers import EarlyStoppingCallback
from transformers.trainer_callback import ProgressCallback
import torchaudio  # 替换 librosa
import random
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from torch.cuda.amp import autocast

def setup_logging(log_file):
    """设置日志"""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # 同时输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

def load_configs(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)

class AudioDataset(Dataset):
    """音频数据集类"""
    def __init__(self, data_dir, processor, model, max_length=48000, mask_prob=0.065):
        self.data_dir = data_dir
        self.processor = processor
        self.model = model  # 添加模型引用
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        self.sampling_rate = 16000  # 添加采样率属性
        
    def __len__(self):
        return len(self.audio_files)
    
    def pad_or_truncate(self, waveform):
        """将音频填充或截断到指定长度"""
        if waveform.shape[-1] > self.max_length:
            start = (waveform.shape[-1] - self.max_length) // 2
            waveform = waveform[start:start + self.max_length]
        elif waveform.shape[-1] < self.max_length:
            padding = self.max_length - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform
    
    def __getitem__(self, idx):
        audio_file = os.path.join(self.data_dir, self.audio_files[idx])
        try:
            waveform, sample_rate = torchaudio.load(audio_file)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0)
            else:
                waveform = waveform.squeeze(0)

            if sample_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
                waveform = resampler(waveform)

            waveform = self.pad_or_truncate(waveform)

            # 处理输入值
            input_values = self.processor(
                waveform, 
                sampling_rate=self.sampling_rate,  # 明确指定采样率
                return_tensors="pt", 
                padding=False
            ).input_values
            
            # 生成掩码索引 (对于预训练必要)
            mask_indices_seq_length = self.model._get_feat_extract_output_lengths(waveform.shape[0])
            mask_time_indices = torch.zeros(mask_indices_seq_length, dtype=torch.bool)
            num_masked = int(mask_indices_seq_length * self.mask_prob)
            indices = random.sample(range(mask_indices_seq_length), num_masked)
            mask_time_indices[indices] = True

            return {
                "input_values": input_values.squeeze(),
                "mask_time_indices": mask_time_indices
            }
            
        except Exception as e:
            logging.error(f"Error loading file {audio_file}: {str(e)}")
            return {"input_values": torch.zeros(self.max_length)}

@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """
    与 Example.py 保持一致的数据整理器
    """
    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.05
    mask_time_length: Optional[int] = 5

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 重新格式化并设置为pytorch格式
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        # 确保掩码序列长度是Python标量
        mask_indices_seq_length = int(mask_indices_seq_length)

        # 确保在填充输入上不计算损失
        if batch.get("attention_mask") is not None:
            # 根据卷积公式计算实际输出长度
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # 随机采样掩码索引
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.mask_time_prob,
            self.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # 采样负样本索引
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch

def multiply_grads(params, c):
    """乘以梯度常数"""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)

def get_grad_norm(params, scale=1):
    """计算梯度范数"""
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm

def freeze_wav2vec2_layers(model, num_layers_to_unfreeze=4):
    """冻结Wav2Vec2模型的层，只保留顶部指定数量的层未冻结
    
    Args:
        model: Wav2Vec2ForPreTraining模型
        num_layers_to_unfreeze: 要保持解冻的Transformer层数量（从顶部开始）
    """
    # 冻结特征提取器
    for param in model.wav2vec2.feature_extractor.parameters():
        param.requires_grad = False
    logging.info("特征提取器已冻结")
    
    # 冻结特征投影层
    for param in model.wav2vec2.feature_projection.parameters():
        param.requires_grad = False
    logging.info("特征投影层已冻结")
    
    # 计算要冻结的Transformer层数
    total_transformer_layers = len(model.wav2vec2.encoder.layers)
    layers_to_freeze = total_transformer_layers - min(num_layers_to_unfreeze, total_transformer_layers)
    
    # 冻结底部的Transformer层
    for i in range(layers_to_freeze):
        for param in model.wav2vec2.encoder.layers[i].parameters():
            param.requires_grad = False
    
    logging.info(f"已冻结底部{layers_to_freeze}个Transformer层")
    
    # 确保顶部Transformer层可训练
    for i in range(layers_to_freeze, total_transformer_layers):
        for param in model.wav2vec2.encoder.layers[i].parameters():
            param.requires_grad = True
    
    # 确保LayerNorm可训练
    for param in model.wav2vec2.encoder.layer_norm.parameters():
        param.requires_grad = True
        
    # 确保量化器组件可训练
    for param in model.project_hid.parameters():
        param.requires_grad = True
    for param in model.project_q.parameters():
        param.requires_grad = True
    for param in model.quantizer.parameters():
        param.requires_grad = True
    
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
    
    # 返回已冻结的层数信息
    return {
        "frozen_feature_extractor": True,
        "frozen_feature_projection": True,
        "frozen_transformer_layers": layers_to_freeze,
        "total_transformer_layers": total_transformer_layers,
        "unfrozen_transformer_layers": total_transformer_layers - layers_to_freeze,
        "trainable_parameters_percentage": trainable_params/total_params
    }

def setup_optimizer_with_layer_specific_lr(model, base_lr, lr_multiplier, num_unfrozen_layers, weight_decay):
    """为不同层设置不同学习率的优化器"""
    encoder_layers = model.wav2vec2.encoder.layers
    total_layers = len(encoder_layers)
    
    # 获取模型中需要优化的参数组
    optimizer_grouped_parameters = []
    
    # 为解冻的encoder层创建参数组（使用递减的学习率）
    for i in range(total_layers - num_unfrozen_layers, total_layers):
        layer_lr = base_lr * (lr_multiplier ** (total_layers - i - 1))
        optimizer_grouped_parameters.append({
            "params": [p for p in encoder_layers[i].parameters() if p.requires_grad],
            "lr": layer_lr,
            "weight_decay": weight_decay
        })
    
    # 为其他可训练参数创建参数组（使用基础学习率）
    other_params = [
        p for n, p in model.named_parameters() 
        if p.requires_grad and not any(f"encoder.layers.{i}." in n 
                                     for i in range(total_layers - num_unfrozen_layers, total_layers))
    ]
    
    optimizer_grouped_parameters.append({
        "params": other_params,
        "lr": base_lr,
        "weight_decay": weight_decay
    })
    
    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        betas=[0.9, 0.999],
        eps=1e-8
    )

class CustomProgressCallback(ProgressCallback):
    """自定义进度条回调"""
    def __init__(self):
        super().__init__()
        self.training_bar = None
        self.last_log = {}
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.training_bar = tqdm(total=state.max_steps, desc="Training")
        
    def on_step_end(self, args, state, control, **kwargs):
        if self.training_bar is not None:
            self.training_bar.update(1)
            if state.log_history:
                self.last_log = state.log_history[-1]
                self.training_bar.set_postfix({
                    'loss': f"{self.last_log.get('loss', 'N/A'):.4f}",
                    'lr': f"{self.last_log.get('learning_rate', 'N/A'):.2e}"
                })
            
    def on_train_end(self, args, state, control, **kwargs):
        if self.training_bar is not None:
            self.training_bar.close()
                
class Wav2Vec2Trainer(Trainer):
    """自定义Trainer类，专门用于Wav2Vec2微调"""
    def __init__(self, *args, **kwargs):
        # 在调用父类构造函数前，先从kwargs中弹出自定义参数
        self.unfreeze_schedule = kwargs.pop("unfreeze_schedule", None)
        self.use_amp = kwargs.pop("use_amp", False)
        
        # 先调用父类构造函数
        super().__init__(*args, **kwargs)
        
        # 然后初始化自定义属性
        self.current_epoch = 0
        # 确保CUDA可用时才启用自动混合精度
        self.use_amp = self.use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # 确保输入在正确设备上并且格式正确
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.device)
        
        outputs = model(**inputs)
        loss = outputs.loss
        
        # 添加损失组件日志
        if hasattr(outputs, "contrastive_loss") and hasattr(outputs, "diversity_loss"):
            try:
                contrastive = outputs.contrastive_loss.item()
                diversity = outputs.diversity_loss.item()
                print(f"Contrastive loss: {contrastive:.2f}, Diversity loss: {diversity:.2f}")
            except Exception as e:
                print(f"记录损失组件时发生错误: {e}")
        
        return (loss, outputs) if return_outputs else loss
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        """支持fp16的训练步骤"""
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            if self.use_amp:
                with autocast():
                    loss = self.compute_loss(model, inputs)
            else:
                loss = self.compute_loss(model, inputs)
        
        # 确保损失不为None
        if loss is None:
            logging.warning("损失为None，使用替代值")
            loss = torch.tensor(1.0, device=model.device, requires_grad=True)
            
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        # 处理反向传播，确保self.scaler不为None
        if self.use_amp and self.scaler is not None:
            try:
                self.scaler.scale(loss).backward()
            except Exception as e:
                logging.error(f"反向传播出错: {str(e)}")
                model.zero_grad()
                return torch.tensor(0.0, device=model.device)
        else:
            try:
                loss.backward()
            except Exception as e:
                logging.error(f"反向传播出错: {str(e)}")
                model.zero_grad()
                return torch.tensor(0.0, device=model.device)
        
        return loss.detach()
    
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time=None):
        """在每个epoch结束时检查是否需要更新解冻的层数"""
        if self.unfreeze_schedule and epoch < len(self.unfreeze_schedule):
            # 更新当前epoch
            self.current_epoch = epoch + 1
            # 检查是否需要解冻更多层
            if epoch + 1 < len(self.unfreeze_schedule):
                # 用下一个epoch的配置更新模型
                stage = self.unfreeze_schedule[epoch + 1]
                num_layers = stage.get("num_layers", 0)
                logging.info(f"进入epoch {epoch + 1}，解冻 {num_layers} 层")
                # 应用新的层冻结配置
                freeze_wav2vec2_layers(model, num_layers)
                
                # 如果学习率调整配置可用，更新优化器
                if "lr_multiplier" in stage:
                    lr_multiplier = stage["lr_multiplier"]
                    base_lr = self.args.learning_rate
                    # 针对不同层设置不同学习率
                    param_groups = []
                    for i, p in enumerate(model.parameters()):
                        if p.requires_grad:
                            # 这里可以根据参数位置分配不同学习率
                            param_groups.append(p)
                    
                    # 创建新的优化器更新学习率
                    self.optimizer = torch.optim.AdamW(
                        param_groups, 
                        lr=base_lr * lr_multiplier
                    )
                    logging.info(f"更新学习率为基础学习率的 {lr_multiplier} 倍")
        
        # 调用父类方法继续正常的日志记录和评估
        return super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

def train_model(config):
    """训练模型"""
    logging.info(f"Starting training with config: {config}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    try:
        # 验证输入
        if not os.path.exists(config['data_dir']):
            raise ValueError(f"Data directory not found: {config['data_dir']}")
            
        logging.info("Initializing model...")
        # 定义一个掩码概率变量
        mask_prob = 0.05
        model = Wav2Vec2ForPreTraining.from_pretrained(
            'facebook/wav2vec2-base',
            mask_time_prob=mask_prob,
            mask_time_length=5,
            contrastive_logits_temperature=0.05,
            num_negatives=50,
            diversity_loss_weight=0.05
        )
        
        model.train()
        # 使用正确的processor - 这里应该使用FeatureExtractor而不是Processor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')
        model = model.to(device)
        
        if config.get('gradient_checkpointing', False):
            model.gradient_checkpointing_enable()
        
        # 读取解冻阶段配置
        unfreeze_stages = config.get('unfreeze_stages', [])
        if not unfreeze_stages:
            logging.warning("未找到unfreeze_stages配置，将使用默认配置")
            unfreeze_stages = [
                {"epochs": 1, "num_layers": 0, "lr_multiplier": 1.0},  # 首先只训练顶层
                {"epochs": 1, "num_layers": 2, "lr_multiplier": 0.5}   # 然后解冻两层
            ]
        
        # 初始解冻配置 - 从第一个阶段开始
        initial_unfrozen_layers = unfreeze_stages[0].get("num_layers", 0)
        freeze_wav2vec2_layers(model, initial_unfrozen_layers)
        
        logging.info("Preparing datasets...")
        train_dataset = AudioDataset(config['data_dir'], feature_extractor, model, max_length=48000)
        val_dataset = AudioDataset(config['data_dir'], feature_extractor, model, max_length=48000)
        
        data_collator = DataCollatorForWav2Vec2Pretraining(
            model=model,
            feature_extractor=feature_extractor,
            pad_to_multiple_of=8,
            mask_time_prob=mask_prob,
            mask_time_length=5  # 已经修改为与模型初始化一致
        )
        
        # 计算训练配置
        total_epochs = sum(stage.get("epochs", 1) for stage in unfreeze_stages)
        logging.info(f"Total training epochs across all stages: {total_epochs}")
        
        training_args = TrainingArguments(
            output_dir=config['output_dir'],
            per_device_train_batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            num_train_epochs=total_epochs,
            logging_dir=os.path.join(config['output_dir'], 'logs'),
            logging_steps=config['log_interval'],
            save_steps=config['log_interval'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            fp16=config['fp16'],
            weight_decay=config['weight_decay'],
            save_total_limit=1,
            remove_unused_columns=False,
            report_to=[],
            run_name=None,
            gradient_checkpointing=True,  # 启用 gradient_checkpointing
            max_grad_norm=0.5  # 从1.0减小到0.5，进一步限制梯度
        )
        
        trainer = Wav2Vec2Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            unfreeze_schedule=unfreeze_stages,  # 自定义参数
            use_amp=config['fp16']  # 自定义参数
        )

        logging.info("Starting training...")
        train_result = trainer.train()
        
        logging.info("Training completed")
        return True, train_result
        
    except Exception as e:
        logging.error(f"Error in train_model: {str(e)}")
        logging.error(traceback.format_exc())
        return False, None

def main():
    parser = argparse.ArgumentParser(description="Train Wav2Vec2 model")
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--test', action='store_true', help='Run test model outputs')
    args = parser.parse_args()
    
    if args.test:
        print("运行模型输出测试...")
        test_model_outputs()
        return
    
    if not args.config:
        parser.error("--config 参数是必需的，除非使用 --test")
    
    # 加载配置
    configs = load_configs(args.config)
    
    # 训练结果统计
    results = {'success': 0, 'failed': 0}
    
    # 遍历所有配置进行训练
    for i, config in enumerate(configs):
        print(f"\nStarting training run: {config['output_dir'].split('run_')[-1]}")
        
        # 验证配置
        required_keys = ['data_dir', 'output_dir', 'batch_size', 'learning_rate', 'total_epochs']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            print(f"Error: Missing required keys in config: {missing_keys}")
            results['failed'] += 1
            continue
            
        # 验证数据目录
        if not os.path.exists(config['data_dir']):
            print(f"Error: Data directory not found: {config['data_dir']}")
            results['failed'] += 1
            continue
            
        # 创建输出目录
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # 设置日志
        log_file = os.path.join(config['output_dir'], 'training.log')
        setup_logging(log_file)
        
        try:
            logging.info(f"Starting configuration {i+1}/{len(configs)}")
            logging.info(f"Config: {config}")
            
            # 开始训练
            success, train_result = train_model(config)
            
            if success:
                logging.info("Training completed successfully")
                results['success'] += 1
                
                # 保存训练结果
                if train_result is not None:
                    result_file = os.path.join(config['output_dir'], 'results.json')
                    with open(result_file, 'w') as f:
                        json.dump({
                            'config': config,
                            'success': True,
                            'training_loss': train_result.training_loss if hasattr(train_result, 'training_loss') else None,
                            'global_step': train_result.global_step if hasattr(train_result, 'global_step') else None
                        }, f, indent=2)
            else:
                logging.error("Training failed")
                results['failed'] += 1
                
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logging.error(f"Unexpected error during training: {str(e)}")
            logging.error(traceback.format_exc())
            results['failed'] += 1
            
        # 等待一会儿确保资源释放
        time.sleep(5)
    
    # 打印最终结果
    print("\nTraining Results Summary:")
    print(f"Total configurations: {len(configs)}")
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")

def test_model_outputs(model=None):
    """测试模型输出结构"""
    if model is None:
        # 移除不支持的参数
        model = Wav2Vec2ForPreTraining.from_pretrained(
            'facebook/wav2vec2-base',
            mask_time_prob=0.05,
            mask_time_length=10,
            contrastive_logits_temperature=0.05,
            num_negatives=50,
            diversity_loss_weight=0.05
        )
    
    # 创建测试批次
    small_batch = {"input_values": torch.randn(2, 16000), 
                  "mask_time_indices": torch.ones(2, 1000, dtype=torch.bool)}
    
    # 执行前向传播
    outputs = model(**small_batch)
    print("输出字段:", outputs.keys() if hasattr(outputs, "keys") else dir(outputs))
    
    # 检查损失相关属性
    for attr in ['loss', 'contrastive_loss', 'diversity_loss']:
        if hasattr(outputs, attr):
            print(f"{attr}: {getattr(outputs, attr)}")
    
    return outputs

if __name__ == "__main__":
    main() 