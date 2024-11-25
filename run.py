import streamlit as st
from src.app import PDFApp
from src.utils import list_model
import tempfile
import os
import webbrowser
from pyvis.network import Network
import numpy as np

# 替换原有的图谱可视化代码
def create_knowledge_graph(query, results, query_embedding, chunk_embeddings):
    # 创建网络图实例
    net = Network(height="700px", width="100%", bgcolor="transparent", font_color="#2c3e50")
    
    # 添加查询节点
    net.add_node(0, label="查询: " + query[:20] + "...", 
                 color="#e74c3c", size=25,
                 shape='dot',
                 font={'size': 16, 'face': 'Arial'})
    
    # 计算所有嵌入之间的相似度矩阵
    chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    
    # 计算查询与文档块的相似度
    query_similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
    
    # 计算文档块之间的相似度矩阵
    chunk_similarities = np.dot(chunk_embeddings, chunk_embeddings.T)
    
    # 添加文档块节点
    for i, (chunk, sim) in enumerate(zip(results, query_similarities), 1):
        net.add_node(i, 
                     label=f"文档块 {i}\n{chunk[:30]}...", 
                     color="#3498db",
                     size=20,
                     shape='dot',
                     font={'size': 14, 'face': 'Arial'})
        
        # 添加与查询节点的边
        width = float(sim) * 8  # 增加边的宽度
        net.add_edge(0, i, width=width, 
                    title=f"相似度: {sim:.3f}",
                    color='rgba(231, 76, 60, 0.5)')
    
    # 为每个文档块节点添加与其他节点的边（保留前5个最相似的）
    for i in range(len(results)):
        # 获取当前节点与其他节点的相似度
        similarities = chunk_similarities[i]
        # 创建索引-相似度对，并排除自身
        pairs = [(j, s) for j, s in enumerate(similarities) if j != i]
        # 按相似度排序并获取前5个
        top_5 = sorted(pairs, key=lambda x: x[1], reverse=True)[:5]
        
        # 添加边
        for j, sim in top_5:
            if i < j:  # 避免重复添加边
                width = float(sim) * 5
                net.add_edge(i+1, j+1, 
                            width=width,
                            title=f"相似度: {sim:.3f}",
                            color='rgba(52, 152, 219, 0.3)')
    
    # 优化物理布局参数
    net.set_options("""
    var options = {
        "nodes": {
            "font": {
                "strokeWidth": 2,
                "strokeColor": "#ffffff"
            }
        },
        "edges": {
            "smooth": false,
            "width": 2,
            "color": {
                "inherit": false
            },
            "shadow": false
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -150,
                "centralGravity": 0.01,
                "springLength": 250,
                "springConstant": 0.08,
                "avoidOverlap": 1
            },
            "maxVelocity": 25,
            "solver": "forceAtlas2Based",
            "timestep": 0.3,
            "stabilization": {
                "enabled": true,
                "iterations": 250,
                "updateInterval": 25
            }
        },
        "interaction": {
            "hover": true,
            "zoomView": true,
            "dragView": true
        },
        "configure": {
            "enabled": false
        },
        "canvas": {
            "background": "transparent"
        }
    }
    """)
    
    # 生成临时HTML文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
        net.save_graph(f.name)
        return f.name

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

        # 添加知识图谱
        st.subheader("知识图谱：")
        query_embedding = st.session_state.pdf_app.querier.get_embedding(query)
        chunk_idxs = st.session_state.pdf_app.querier.find_similar_chunks(query_embedding, st.session_state.pdf_app.embeddings, top_k)
        chunk_embeddings = [st.session_state.pdf_app.embeddings[idx] for idx in chunk_idxs]
        
        # 生成图谱
        html_path = create_knowledge_graph(query, results, query_embedding, chunk_embeddings)
        
        # 使用 HTML 组件显示图谱
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600)

# 添加使用说明
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    ### 使用说明
    1. 在侧边栏配置参数
    2. 上传 PDF 文件
    3. 输入查询内容
    4. 回车查看搜索结果
    """)

# streamlit run run.py
