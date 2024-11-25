from ollama import Client
from tqdm import tqdm
from ._types import EmbeddingResponse
from typing import List

class Embedder:
    def __init__(self, host: str, embed_model: str, batch_size: int):
        self.client = Client(host)
        self.model = embed_model
        self.batch_size = batch_size

    def embed(self, chunks: List[str]) -> List[List[float]]:
        embeddings = []
        for i in tqdm(range(0, len(chunks), self.batch_size)):
            chunks_batch = chunks[i:i + self.batch_size]
            response = self.client.embed(self.model, chunks_batch)
            pack_data = EmbeddingResponse.from_dict(response)
            embeddings.extend(pack_data.embeddings)
        assert len(chunks) == len(embeddings), \
            "Number of chunks and embeddings do not match."
        return embeddings

if __name__ == '__main__':
    import numpy as np
    embedder = Embedder(
        host='http://localhost:11434', 
        # embed_model='nomic-embed-text', # 768
        embed_model='all-minilm', # 384
        batch_size=10
    )
    chunks = ['Hello, world!', 'This is a test.']
    embeddings = embedder.embed(chunks)
    print(np.array(embeddings).shape)

# python -m src.embed