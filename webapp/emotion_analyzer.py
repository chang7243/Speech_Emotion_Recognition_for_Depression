import streamlit as st
from components.visualizations import plot_emotion_distribution
from utils import model_inference
from components.audio_player import play_audio
from components.debug_tools import DebugTools
import json
import os
import time

# ✅ 初始化 session_state 以存储分析结果和反馈状态
if "emotion_result" not in st.session_state:
    st.session_state.emotion_result = None
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
if "user_feedback" not in st.session_state:
    st.session_state.user_feedback = ""

def show_history():
    """📜 显示历史记录（最近 5 条）"""
    history_file = "history/history.json"
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        if history:
            st.subheader("📜 History")
            with st.expander("Click to view past records", expanded=False):
                for record in history[-5:]:  # 只显示最近 5 条
                    st.write(f"🎧 **Audio File**: {record['audio_file']}")
                    st.write(f"📝 **Transcript**: {record['transcript']}")
                    st.write(f"🎭 **Emotions**: {record['emotions']}")
                    st.write("---")
        else:
            st.write("🚫 No history records found.")
    else:
        st.write("🚫 No history file.")

def show():
    # ✅ 初始化 session_state 以存储分析结果和反馈状态
    if "emotion_result" not in st.session_state:
        st.session_state.emotion_result = None
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False
    if "user_feedback" not in st.session_state:
        st.session_state.user_feedback = ""
    st.markdown(
        "<h1 style='text-align: center; color: #FF5733;'>😉 Emotion Analyzer 🎭</h1>",
        unsafe_allow_html=True
    )

    # 📜 显示历史记录
    show_history()
    
    # 🛠️ 初始化调试工具
    debug = DebugTools()
    debug.show_debug_info()
    
    # 🎤 让用户上传音频和文本输入
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        audio_file = st.file_uploader("Upload your audio file", type=['wav', 'mp3'])
    # 🎧 如果音频存在，则显示音频信息
        if audio_file is not None:
            debug.show_audio_info(audio_file)
            st.audio(audio_file, format="audio/mp3")
    with col2:
        text_input = st.text_input("Enter your text input")
        transcript = text_input  # 默认情况下，文本输入即为转写结果
        st.write("📄 **Audio transcript:**", transcript)

    # 🎯 **如果用户删除了音频，则自动重置情绪分析结果**
    if audio_file is None and "emotion_result" in st.session_state and st.session_state.emotion_result is not None:
        st.warning("🚫 Audio file removed. Resetting analysis results.")
        st.session_state.emotion_result = None

    # # 🎧 如果音频存在，则显示音频信息
    # if audio_file is not None:
    #     debug.show_audio_info(audio_file)
    #     st.audio(audio_file, format="audio/mp3")
    
    if audio_file is not None and text_input:
        if st.button("Analyse Your Emotion! 😊"):
            st.session_state.feedback_submitted = False  # 重置反馈状态

            # 📊 显示进度条
            progress_bar = st.progress(0)
            for percent in range(0, 101, 20):
                progress_bar.progress(percent)
                time.sleep(0.2)  # 提高加载体验

            try:
                transcript = text_input

                # 🔍 执行情绪分析
                with st.spinner("Analysing emotion..."):
                    for percent in range(30, 101, 20):  
                        progress_bar.progress(percent)
                        time.sleep(0.2)  # 避免过快闪过
                    emotions = model_inference.predict_emotion(text_input, audio_file)

                if emotions:
                    best_emotion = max(emotions, key=lambda x: float(emotions[x].strip('%')))
                    # 🎭 显示情绪分析结果
                    st.markdown("---")
                    st.markdown("### 🎭 Emotion Analysis Results")
                    st.success(f"🎭 **Predicted Emotion:** {best_emotion}")
                    st.success(f"📊 **Emotion Probabilities:** {emotions}")

                    # ✅ 存储分析结果，防止页面刷新导致结果丢失
                    st.session_state.emotion_result = {
                        "best_emotion": best_emotion,
                        "emotions": emotions,
                        "transcript": transcript
                    }

            except Exception as e:
                debug.log_error(e, context=f"Processing file: {audio_file.name}")

    # 🎭 只有当有分析结果时，才显示结果
    if st.session_state.emotion_result:
        best_emotion = st.session_state.emotion_result["best_emotion"]
        emotions = st.session_state.emotion_result["emotions"]

        # 🎈 Streamlit 内置动画（仅用于 "Joy" 和 "Sad"）
        if best_emotion == "Joy":
            st.balloons()
            st.markdown("**Yay! You seem so happy! 🎉😊**")
            st.write("Keep smiling, it's contagious! 🌟")
        elif best_emotion == "Sad":
            st.snow()
            st.markdown("**Oh no, you seem a bit down... 😔💭**")
            st.write("I'm here if you need to talk or cheer up! 💬💖")
        else:
            emotion_messages = {
                "Angry": ["😡 **Whoa! Looks like you're really angry...**", "Take a deep breath, let's cool off together! 🧘‍♂️"],
                "Surprised": ["🤯 **Wow, you're really surprised!**", "That's a big reaction! Hope it's a good surprise! 🎉"],
                "Fearful": ["😨 **Looks like you're feeling scared...**", "It's okay to be scared. I'm here to help! 💪"],
                "Disgusted": ["🤢 **Yikes, you're feeling disgusted...**", "That doesn't sound pleasant. Let's turn it around! 🌈"],
                "Neutral": ["🤔 **Hmm, your emotions seem mixed...**", "Can you tell me more? Let's figure this out together! 💬"]
            }
            if best_emotion in emotion_messages:
                st.markdown(emotion_messages[best_emotion][0])
                st.write(emotion_messages[best_emotion][1])

        # 📊 显示情绪分布图
        plot_emotion_distribution(emotions)
    
        # 💾 保存历史记录
        if audio_file is not None:
            model_inference.save_history(audio_file, transcript, emotions)
        else:
            st.warning("⚠️ No audio file uploaded. History not saved.")

    # 📣 提供反馈功能
    if st.session_state.emotion_result:
        st.subheader("📣 Give us your feedback!")
        feedback = st.radio("Do you agree with the prediction?", ["😀 Great!", "😐 Okay", "😡 Wrong"], index=1)
        user_feedback = st.text_area("Tell us more (optional):", st.session_state.user_feedback)

        if st.button("Submit Feedback"):
            st.session_state.user_feedback = user_feedback
            st.session_state.feedback_submitted = True

        if st.session_state.feedback_submitted:
            st.success("✅ Thank you for your feedback! It has been recorded.")


