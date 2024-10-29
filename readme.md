# PDFapp

这里希望做一个简单的PDF查询检索的APP, 首先要做的事情是:

1. 从PDF中提取文本chunk 建好索引 存储到本地数据库中
2. 将每个chunk分batch传到服务器 拿回embedding
3. 将embedding保存到数据库中

## 技术方案

- 数据库: 打算直接使用csv文件存储
- 服务器: ollama的API
- 语言: python

## app 环境配置

> 这里单独使用venv来配置环境，方便到时候打包软件

```shell
python -m venv venv
source venv/bin/activate # windows下使用venv\Scripts\activate
# pip install -r requirements.txt
python -m pip install --upgrade pip
pip install streamlit PyMuPDF
```

## reference

- [pdf.js](https://github.com/mozilla/pdf.js)
