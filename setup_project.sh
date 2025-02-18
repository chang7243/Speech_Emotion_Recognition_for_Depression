#!/bin/bash

# 创建项目主目录
mkdir -p emotion_recognition_project/{data/raw,data/processed,data/pseudo_labels,data/splits}
mkdir -p emotion_recognition_project/{models/wav2vec2,models/bert,models/multimodal}
mkdir -p emotion_recognition_project/{experiments/logs,experiments/results,experiments/figures}
mkdir -p emotion_recognition_project/src/{preprocess,ssl_training,supervised_training,pseudo_labeling,multimodal,evaluation}
mkdir -p emotion_recognition_project/notebooks

# 创建关键文件
touch emotion_recognition_project/README.md
touch emotion_recognition_project/requirements.txt
touch emotion_recognition_project/train.py
touch emotion_recognition_project/config.yaml

echo "✅ 项目结构已创建！"
