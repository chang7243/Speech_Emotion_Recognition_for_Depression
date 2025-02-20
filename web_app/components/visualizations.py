import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def plot_emotion_distribution(emotions, probabilities):
    """
    绘制情绪概率分布图
    
    Args:
        emotions: 情绪标签列表
        probabilities: 对应的概率列表
    """
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=probabilities,
            marker_color='rgb(26, 118, 255)'
        )
    ])
    
    fig.update_layout(
        title="情绪概率分布",
        xaxis_title="情绪类别",
        yaxis_title="概率",
        yaxis_range=[0, 1]
    )
    
    st.plotly_chart(fig, use_container_width=True) 