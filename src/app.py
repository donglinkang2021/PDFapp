from .chunk import Chunker
from .embed import Embedder
from .query import Querier
from pathlib import Path
import pandas as pd

class PDFApp:
    def __init__(self, host: str, embed_model: str, chunk_size: int, batch_size: int):
        self.chunker = Chunker(chunk_size)
        self.embedder = Embedder(host, embed_model, batch_size)
        self.querier = Querier(host, embed_model)

    def load_pdf(self, pdf_path: str):
        self.chunks = self.chunker.chunk(pdf_path)
        self.embeddings = self.embedder.embed(self.chunks)

    def query(self, query: str, top_k: int = 10):
        query_embedding = self.querier.get_embedding(query)
        similar_indices = self.querier.find_similar_chunks(query_embedding, self.embeddings, top_k)
        return [self.chunks[idx] for idx in similar_indices]

    def save(self, save_dir: str):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model_df = pd.DataFrame({
            'chunks': self.chunks,
            'embeddings': self.embeddings
        })
        model_df.to_csv(save_dir / 'model.csv', index=False)

    def load(self, load_dir: str):
        load_dir = Path(load_dir)
        model_df = pd.read_csv(load_dir / 'model.csv')
        self.chunks = model_df['chunks'].tolist()
        embeddings = model_df['embeddings'].tolist()
        self.embeddings = [eval(embed) for embed in embeddings]


def preprocess_pdf(
        pdf_path: str, 
        save_dir: str, 
        host: str, 
        embed_model: str, 
        chunk_size: int, 
        batch_size: int
    ):
    app = PDFApp(host, embed_model, chunk_size, batch_size)
    app.load_pdf(pdf_path)
    app.save(save_dir)
    print(f"Model saved to {save_dir}.")

def query_pdf(
        load_dir: str, 
        host: str, 
        embed_model: str, 
        query: str, 
        top_k: int = 10
    ):
    app = PDFApp(host, embed_model, 0, 0)
    app.load(load_dir)
    return app.query(query, top_k)

def main():
    # 读取 PDF 文件并预处理
    pdf_path = 'data/temp.pdf'  # 替换为你的 PDF 文件路径
    save_dir = 'data/model'  # 保存模型的目录
    host = 'http://localhost:11434'
    embed_model = 'all-minilm'
    chunk_size = 500
    batch_size = 10
    preprocess_pdf(pdf_path, save_dir, host, embed_model, chunk_size, batch_size)

    # 查询 PDF 文件
    load_dir = 'data/model'  # 加载模型的目录
    query = 'Machine learning is fun.'  # 查询关键词
    top_k = 5
    similar_chunks = query_pdf(load_dir, host, embed_model, query, top_k)
    print(f"Top {top_k} similar chunks to '{query}':")
    for chunk in similar_chunks:
        print(chunk)

if __name__ == '__main__':
    main()

# python -m src.app
