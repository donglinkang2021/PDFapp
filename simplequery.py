import pandas as pd
import numpy as np
from ollama import Client
from typing import List, Union

client = Client(host='http://localhost:6837')

def get_embedding_from_server(query:Union[str, List[str]]) -> np.ndarray:
    response = client.embed(model='all-minilm', input=query)
    return np.array(response['embeddings'], dtype=np.float32)

def find_similar_chunks(
        query_embedding:np.ndarray, 
        embeddings:np.ndarray, 
        original_data:pd.DataFrame, 
        top_n:int=10
    ):
    # similarities 
    similarities:np.ndarray = np.dot(embeddings, query_embedding.T)
    # top_indices
    top_indices = similarities.argsort(axis=0)[-top_n:][::-1].reshape(-1)
    return original_data.iloc[top_indices]

def main():
    # 读取 embeddings 和原始文本
    embedd_df = pd.read_csv('PDFapp/data/embedd.csv')
    chunk_df = pd.read_csv('PDFapp/data/chunk.csv')

    # 提取 embeddings 和原文片段
    embeddings = np.array([eval(embed) for embed in embedd_df['embed']])  # 将字符串转换为数组
    original_data = chunk_df.set_index('id')  # 设置索引为 ID

    # 输入关键词
    query = input("请输入关键词：")
    
    # 获取查询的 embedding
    print("请求服务器获取查询的 embedding...")
    query_embedding = get_embedding_from_server(query)

    # 查找相似文本片段
    print("查找相似文本片段...")
    similar_chunks = find_similar_chunks(query_embedding, embeddings, original_data)

    # 显示结果
    print("相似度最高的前十个片段：")
    for idx, row in similar_chunks.iterrows():
        print(f"ID: {idx}, 片段: {row['chunk']}")

if __name__ == '__main__':
    main()

# python PDFapp\simplequery.py