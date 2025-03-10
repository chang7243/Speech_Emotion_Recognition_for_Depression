import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def plot_emotion_distribution(emotion_dict):
    """
    绘制情绪概率分布图（雷达图）
    
    Args:
        emotion_dict: 包含情绪标签和对应概率的字典
    """
    emotions = list(emotion_dict.keys())
    probabilities = [float(emotion_dict[emotion].strip('%')) / 100 for emotion in emotions]  # 转换为浮点数

    # 创建雷达图
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=probabilities + [probabilities[0]],  # 闭合图形
        theta=emotions + [emotions[0]],  # 闭合图形
        fill='toself',
        name='Emotion Distribution'
    ))

    fig.update_layout(
        title="Emotion Distribution",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # 设置范围
            )),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True) 