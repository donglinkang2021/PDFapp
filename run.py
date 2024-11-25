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
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # 添加查询节点
    net.add_node(0, label="查询: " + query[:20] + "...", color="#ff0000", size=20)
    
    # 计算相似度
    chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
    
    # 添加文档块节点和边
    for i, (chunk, sim) in enumerate(zip(results, similarities), 1):
        # 添加节点
        net.add_node(i, label=f"文档块 {i}\n{chunk[:30]}...", color="#0000ff", size=15)
        
        # 添加边，使用相似度作为边的宽度
        width = float(sim) * 5  # 调整边的宽度
        net.add_edge(0, i, width=width, title=f"相似度: {sim:.2f}")
    
    # 设置物理布局参数
    net.set_options("""
    var options = {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
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
