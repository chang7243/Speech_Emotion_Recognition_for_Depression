import torch
import json
import os
from transformers import AutoTokenizer, BertModel, Wav2Vec2Model
from utils.audio_processing import AudioProcessor
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoConfig, Wav2Vec2ForPreTraining

# 下载模型
# huggingface_hub 仓库下载
# model_path = hf_hub_download(repo_id="liloge/Group7_model_test", filename="model.safetensors")
# 本地下载

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig, Wav2Vec2ForPreTraining

class MultimodalClassifier(nn.Module):
    def __init__(self, wav2vec2_config_path):
        super().__init__()
        
        # **加载微调后的 BERT**
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=7
        )
        self.bert.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.bert.config.hidden_size, self.bert.config.num_labels)
        )
        # try:
        #     self.bert.load_state_dict(torch.load(bert_ckpt_path, map_location=torch.device("cpu")), strict=True)
        # except Exception as e:
        #     print(f"❌ 加载 `{bert_ckpt_path}` 失败: {e}")
            
        # **先加载 Wav2Vec2**
        config = AutoConfig.from_pretrained(wav2vec2_config_path, num_labels=7)
        self.wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base", config=config)

        # **再修改 Wav2Vec2 的分类头**
        self.wav2vec2.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.wav2vec2.config.hidden_size, self.wav2vec2.config.num_labels)
        )
        # # **加载 safetensors 权重**
        # from safetensors.torch import load_file
        # state_dict = load_file(wav2vec2_safetensors_path)
        # try:
        #     self.wav2vec2.load_state_dict(state_dict, strict=False)
        # except Exception as e:
        #     print(f"❌ 加载 `{wav2vec2_safetensors_path}` 失败: {e}")

        # **拼接特征的分类头**
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + self.wav2vec2.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 7)  # 7分类任务
        )

    def forward(self, text_input, audio_input):
        # **文本特征**
        text_outputs = self.bert(**text_input, output_hidden_states=True)
        text_features = text_outputs.hidden_states[-1][:, 0, :]

        # **音频特征**
        audio_outputs = self.wav2vec2(audio_input, output_hidden_states=True)
        audio_features = audio_outputs.hidden_states[-1][:, 0, :]

        # **拼接特征**

        combined_features = torch.cat((text_features, audio_features), dim=-1)

        # **分类**
        logits = self.classifier(combined_features)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **定义路径**
wav2vec2_config_path = r"models/config.json"
model_path = r"models/model.safetensors"

# **加载模型及其参数**
model = MultimodalClassifier(wav2vec2_config_path).to(device)

state_dict = load_file(model_path)
model.load_state_dict(state_dict, strict=True)
model.to(device)
model.eval()

print("✅ 微调的 BERT + Wav2Vec2 模型加载成功！")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_text(text):
    text_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    return text_inputs.to(device)

def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    return waveform.to(device)

labels = ["Neutral", "Joy", "Sad", "Angry", "Surprised", "Fearful", "Disgusted"]

def predict_emotion(text, audio):
    text_inputs = preprocess_text(text)
    audio_inputs = preprocess_audio(audio)

    with torch.no_grad():
        output = model(text_input=text_inputs, audio_input=audio_inputs)  # (1, 7) logits
        probabilities = F.softmax(output, dim=1).squeeze().tolist()  # 归一化为概率

    return {labels[i]: f"{probabilities[i]*100:.2f}%" for i in range(len(labels))}

def generate_transcript(audio_file):
    """生成音频的文字转写"""
    return audio_file.name  # 直接返回音频文件的名称

def save_history(audio_file, transcript, emotions):
    """保存分析历史记录到文件"""
    history_file = r"history/history.json"
    
    if not os.path.exists(history_file):
        with open(history_file, 'w') as f:
            json.dump([], f)
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    history.append({
        "audio_file": audio_file.name,
        "transcript": transcript,
        "emotions": emotions,
    })
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)