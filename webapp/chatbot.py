import streamlit as st
import torch
import torchaudio
import json
from openai import AzureOpenAI
from openai.types.beta.threads import Message
from safetensors.torch import load_file
from transformers import AutoTokenizer, Wav2Vec2Processor, BertModel, Wav2Vec2Model
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from utils import model_inference
import os

# 加载环境变量
load_dotenv(r"Group7/.env")
api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
api_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# 初始化 OpenAI 客户端
client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=api_endpoint)

# 设定 Chatbot 角色
instruction = (
    "You are a psychiatrist talking to a patient who may be depressed. "
    "You'll receive their emotional state and conversation text. "
    "Your goal is to help them open up and guide them to a positive path. "
    "Be friendly, professional, empathetic, and supportive."
)

# 设定 Chatbot 线程和助手
if "thread" not in st.session_state:
    st.session_state.thread = client.beta.threads.create()

if "assistant" not in st.session_state:
    assistant_id = "asst_Sb1W9jVTeL1iyzu6N5MilgA1"
    try:
        st.session_state.assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
    except:
        st.session_state.assistant = client.beta.assistants.create(
            name="Depression Chatbot",
            instructions=instruction,
            model=api_deployment_name,
        )

# 发送消息到 Azure Chatbot
def send_message_to_chatbot(user_input, emotion):
    chat_history = client.beta.threads.messages.list(thread_id=st.session_state.thread.id)
    messages = [{"role": msg.role, "content": msg.content} for msg in chat_history]
    
    messages.append({"role": "user", "content": f"Emotion: {emotion}. {user_input}"})

    client.beta.threads.messages.create(
        thread_id=st.session_state.thread.id,
        role="user",
        content=f"Emotion: {emotion}. {user_input}",
    )

    run = client.beta.threads.runs.create(
        thread_id=st.session_state.thread.id,
        assistant_id=st.session_state.assistant.id,
    )

    while run.status in ["queued", "in_progress"]:
        run = client.beta.threads.runs.retrieve(run.id)

    response_messages = client.beta.threads.messages.list(thread_id=st.session_state.thread.id)
    return response_messages[-1].content if response_messages else "No response."

# Streamlit 界面
st.title("🧠 AI Depression Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 用户输入
user_input = st.text_input("Enter your message:")
audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])

if st.button("Send"):
    if user_input or audio_file:
        emotion_probabilities = model_inference.predict_emotion(user_input, audio_file)
        dominant_emotion = max(emotion_probabilities, key=emotion_probabilities.get)

        chatbot_response = send_message_to_chatbot(user_input, dominant_emotion)
        
        # 保存聊天记录
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": chatbot_response})

        # 显示聊天记录
        for chat in st.session_state.chat_history:
            st.write(f"**{chat['role'].capitalize()}**: {chat['content']}")

    else:
        st.warning("Please enter a message or upload an audio file.")
