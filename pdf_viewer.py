import streamlit as st

# 设置页面布局为宽屏模式并隐藏默认菜单栏
st.set_page_config(layout="wide")

# 在侧边栏添加上传按钮和搜索功能
with st.sidebar:
    pdf_file = st.file_uploader("上传PDF文件", type=["pdf"])

if pdf_file is not None:
    # 保存上传的文件
    with open("data/temp.pdf", "wb") as f:
        f.write(pdf_file.read())

    # 使用 PDF.js 显示 PDF 文件，添加搜索功能支持
    pdf_path = "../../data/temp.pdf"
    pdf_js_path = f"http://localhost:8000/pdfjs/web/viewer.html?file={pdf_path}"
    st.components.v1.html(
        f"""
        <iframe src="{pdf_js_path}" width="100%" height="800px" id="pdf-viewer"></iframe>
        """,
        height=800
    )

# 需要在当前目录同时启动下面两条命令
# python -m http.server
# streamlit run pdf_viewer.py