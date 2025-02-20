import streamlit as st
import os
import sys
from pathlib import Path

# 确保能找到项目模块
sys.path.append(str(Path(__file__).parent.parent))

from pages import emotion_analyzer  # 先只导入已实现的页面

def main():
    st.set_page_config(
        page_title="情绪识别系统",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("音频情绪识别系统")
    
    # 直接显示情绪分析页面
    emotion_analyzer.show()

if __name__ == "__main__":
    main() 