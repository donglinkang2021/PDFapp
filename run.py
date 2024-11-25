import streamlit as st
from src.app import PDFApp
from src.utils import list_model
import tempfile
import os

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