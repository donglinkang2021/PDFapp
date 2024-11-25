import streamlit as st
from src.app import PDFApp
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
    host = st.text_input("Ollama 服务器地址", value="http://localhost:11434")
    embed_model = st.text_input("嵌入模型", value="all-minilm")
    chunk_size = st.number_input("文本块大小", value=500, min_value=100)
    batch_size = st.number_input("批处理大小", value=10, min_value=1)

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
    query = st.text_input("输入查询内容")
    top_k = st.slider("返回结果数量", min_value=1, max_value=20, value=5)

    if query and st.button("搜索"):
        with st.spinner("正在搜索..."):
            results = st.session_state.pdf_app.query(query, top_k)
        
        st.subheader("搜索结果：")
        for i, chunk in enumerate(results, 1):
            with st.expander(f"结果 {i}"):
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
