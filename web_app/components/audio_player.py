import streamlit as st

def play_audio(audio_file):
    """
    显示音频播放器组件
    
    Args:
        audio_file: 上传的音频文件
    """
    st.audio(audio_file) 