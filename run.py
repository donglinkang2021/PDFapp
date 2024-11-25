import streamlit as st
from src.app import PDFApp
from src.utils import list_model
import tempfile
import os
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import numpy as np

st.set_page_config(page_title="PDF 语义搜索", layout="wide")
st.title("PDF 语义搜索应用")

# 初始化 session state
if 'pdf_app' not in st.session_state:
    st.session_state.pdf_app = None

# 侧边栏配置
with st.sidebar:
    st.header("配置参数")
    host = st.text_input("Ollama 服务器地址", value="http://localhost:11434", key="host")
    
    # 获取可用模型列表
    try:
        available_models = list_model(host)
        default_model = "all-minilm:latest"
        embed_model = st.selectbox(
            "嵌入模型", options=available_models, 
            index=available_models.index(default_model) if default_model in available_models else 0, 
            key="embed_model"
        )
    except Exception as e:
        st.error(f"无法连接到 Ollama 服务器: {str(e)}")
        embed_model = st.text_input("嵌入模型", value="None", key="embed_model")
    
    chunk_size = st.number_input("文本块大小", value=200, min_value=100, key="chunk_size")
    batch_size = st.number_input("批处理大小", value=10, min_value=1, key="batch_size")
    
    # 检测参数是否发生变化
    current_params = f"{host}-{embed_model}-{chunk_size}-{batch_size}"
    if 'previous_params' not in st.session_state:
        st.session_state.previous_params = current_params
    
    # 如果参数发生变化，重置 pdf_app
    if current_params != st.session_state.previous_params:
        st.session_state.pdf_app = None
        st.session_state.previous_params = current_params

# 主界面
uploaded_file = st.file_uploader("上传 PDF 文件", type=['pdf'])

if uploaded_file:
    # 创建临时文件保存上传的 PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name

    # 初始化 PDFApp 并处理 PDF
    if st.session_state.pdf_app is None:
        with st.spinner("正在处理 PDF 文件..."):
            st.session_state.pdf_app = PDFApp(host, embed_model, chunk_size, batch_size)
            st.session_state.pdf_app.load_pdf(pdf_path)
        st.success("PDF 处理完成！")

    # 删除临时文件
    os.unlink(pdf_path)

    # 查询界面
    query = st.text_input("输入查询内容", key="query", on_change=None)
    top_k = st.slider("返回结果数量", min_value=1, max_value=20, value=6)

    # 检查是否有查询内容且 pdf_app 已初始化
    if query and st.session_state.pdf_app:
        with st.spinner("正在搜索..."):
            results = st.session_state.pdf_app.query(query, top_k)
        
        st.subheader("搜索结果：")
        for i, chunk in enumerate(results, 1):
            # 创建预览文本（取前80个字符，如果原文更长则添加省略号）
            preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
            with st.expander(f"结果 {i} - {preview}"):
                st.write(chunk)

        # 添加图谱展示
        st.subheader("相似度图谱：")
        
        # 获取所有相关的 embeddings
        query_embedding = st.session_state.pdf_app.querier.get_embedding(query)
        chunk_idxs = st.session_state.pdf_app.querier.find_similar_chunks(query_embedding, st.session_state.pdf_app.embeddings, top_k)
        chunk_embeddings = np.array([st.session_state.pdf_app.embeddings[idx] for idx in chunk_idxs])
        all_embeddings = np.vstack([query_embedding, chunk_embeddings])
        
        # 使用 t-SNE 降维到2D，调整 perplexity 参数
        n_samples = len(all_embeddings)
        perplexity = min(30, n_samples - 1)  # 确保 perplexity 小于样本数
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            max_iter=250
        )
        embeddings_2d = tsne.fit_transform(all_embeddings)
        
        # 创建图谱
        fig = go.Figure()
        
        # 添加查询点
        fig.add_trace(go.Scatter(
            x=[embeddings_2d[0][0]],
            y=[embeddings_2d[0][1]],
            mode='markers+text',
            marker=dict(size=15, color='red'),
            text=['查询'],
            textposition="top center",
            name='查询'
        ))
        
        # 添加文档块点
        fig.add_trace(go.Scatter(
            x=embeddings_2d[1:, 0],
            y=embeddings_2d[1:, 1],
            mode='markers+text',
            marker=dict(size=10, color='blue'),
            text=[f'结果 {i+1}' for i in range(len(results))],
            textposition="top center",
            name='文档块'
        ))
        
        # 更新布局
        fig.update_layout(
            title="查询结果相似度可视化",
            showlegend=True,
            width=800,
            height=600
        )
        
        st.plotly_chart(fig)

# 添加使用说明
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    ### 使用说明
    1. 在侧边栏配置参数
    2. 上传 PDF 文件
    3. 输入查询内容
    4. 点击搜索按钮查看结果
    """)

# streamlit run run.py