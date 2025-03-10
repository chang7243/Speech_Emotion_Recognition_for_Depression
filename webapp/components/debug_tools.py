import streamlit as st
import sys
import os
import platform
from datetime import datetime

class DebugTools:
    @staticmethod
    def show_debug_info():
        """显示调试信息的可折叠部分"""
        with st.expander("Debug Information", expanded=False):
            # 系统信息
            st.subheader("System Information")
            st.text(f"System: {platform.system()} {platform.version()}")
            st.text(f"Python Version: {sys.version}")
            
            # 内存使用
            try:
                import psutil
                process = psutil.Process(os.getpid())
                st.text(f"Memory Usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            except ImportError:
                st.text("Memory Usage: Unable to get (requires psutil)")
            
            # GPU信息
            try:
                import torch
                if torch.cuda.is_available():
                    st.text(f"GPU: {torch.cuda.get_device_name(0)}")
                    st.text(f"GPU Memory: {torch.cuda.memory_allocated(0)/1024/1024:.2f}MB / "
                           f"{torch.cuda.memory_reserved(0)/1024/1024:.2f}MB")
                else:
                    st.text("GPU: Not Available")
            except Exception as e:
                st.text("GPU Information Retrieval Failed")
    
    @staticmethod
    def log_error(error, context=None):
        """记录错误信息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_msg = f"[{timestamp}] Error: {str(error)}"
        if context:
            error_msg += f"\nContext: {context}"
        
        st.error(error_msg)
        # 可以添加日志文件记录
        print(error_msg, file=sys.stderr)
    
    @staticmethod
    def show_audio_info(audio_file):
        """显示音频文件信息"""
        if audio_file is not None:
            st.write("Audio File Information:")
            st.text(f"File Name: {audio_file.name}")
            st.text(f"File Size: {audio_file.size/1024:.2f} KB")
            st.text(f"File Type: {audio_file.type}") 