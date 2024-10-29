import pandas as pd
from ollama import Client
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

@dataclass
class EmbeddingResponse:
    model: str
    embeddings: np.ndarray[np.float32]
    total_duration: int = None
    load_duration: int = None
    prompt_eval_count: int = None

    @staticmethod
    def from_dict(data: dict) -> 'EmbeddingResponse':
        return EmbeddingResponse(
            model=data['model'],
            embeddings=np.array(data['embeddings'], dtype=np.float32),
            total_duration=data.get('total_duration'),
            load_duration=data.get('load_duration'),
            prompt_eval_count=data.get('prompt_eval_count')
        )

client = Client(host='http://localhost:6837')

def get_embeddings(chunks, batch_size):
    embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        response = client.embed(model='all-minilm', input=batch)
        pack_data = EmbeddingResponse.from_dict(response)
        embeddings.extend(pack_data.embeddings.tolist())
    return embeddings

def main():
    # 读取 CSV 文件
    csv_path = 'PDFapp/data/chunk.csv'  # 替换为你的 chunk.csv 文件路径
    data = pd.read_csv(csv_path)

    # 设置批量大小
    batch_size = 10  # 根据需要调整批量大小

    # 获取 embeddings
    print("Fetching embeddings...")
    embeddings = get_embeddings(data['chunk'].tolist(), batch_size)

    # 保存到 CSV 文件
    embedd_df = pd.DataFrame({
        'id': data['id'],
        'embed': embeddings
    })
    
    embedd_df.to_csv('PDFapp/data/embedd.csv', index=False)
    print("Embeddings saved to embedd.csv.")

if __name__ == '__main__':
    main()

# python PDFapp\chunk2embeddings.py