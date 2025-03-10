import asyncio
import sys
import streamlit as st
import os
from pathlib import Path
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

sys.path.append(str(Path(__file__).parent))

def main():
    st.set_page_config(
        page_title="Audio Emotion Recognition System",
        page_icon="🎵",
        layout="wide"
    )
    st.sidebar.title("Navigation Bar")
    st.sidebar.markdown("<small>(Chatbot is not available now, update soon...😉)</small>", unsafe_allow_html=True)
    app_mode = st.sidebar.radio("Go to", ["Emotion Analyzer", "Chatbot"])
    if app_mode == "Emotion Analyzer":
        from emotion_analyzer import show  # 直接导入模块
        # ✅ 在 `main.py` 里初始化 `session_state`
        if "emotion_result" not in st.session_state:
            st.session_state.emotion_result = None
        if "feedback_submitted" not in st.session_state:
            st.session_state.feedback_submitted = False
        if "user_feedback" not in st.session_state:
            st.session_state.user_feedback = ""
        show()
    elif app_mode == "Chatbot":
        st.write("Chatbot is not available now, update soon...😉")

if __name__ == "__main__":
    main() 