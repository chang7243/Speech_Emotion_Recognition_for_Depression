import streamlit as st
from components.visualizations import plot_emotion_distribution
from utils import model_inference
import numpy as np
from components.audio_player import play_audio
from components.debug_tools import DebugTools

def show():
    st.header("情绪分析")
    
    # 初始化调试工具
    debug = DebugTools()
    
    # 显示系统调试信息
    debug.show_debug_info()
    
    # 文件上传
    audio_file = st.file_uploader("上传音频文件", type=['wav', 'mp3'])
    
    if audio_file is not None:
        # 显示音频文件信息
        debug.show_audio_info(audio_file)
        
        # 使用audio_player组件
        play_audio(audio_file)
        
        if st.button("开始分析"):
            # 显示进度条
            progress_bar = st.progress(0)
            
            try:
                # 1. 音频转写
                with st.spinner("正在转写音频..."):
                    progress_bar.progress(30)
                    transcript = model_inference.generate_transcript(audio_file)
                    st.write("音频转写结果:", transcript)
                
                # 2. 情绪分析
                with st.spinner("正在分析情绪..."):
                    progress_bar.progress(60)
                    emotions, probabilities = model_inference.predict_emotion(audio_file, transcript)
                
                # 3. 显示结果
                progress_bar.progress(100)
                
                # 显示预测结果
                st.success(f"预测情绪: {emotions}")
                
                # 显示情绪概率分布图
                plot_emotion_distribution(emotions, probabilities)
                
            except Exception as e:
                debug.log_error(e, context=f"处理文件: {audio_file.name}") 