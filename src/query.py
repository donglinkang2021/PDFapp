import numpy as np
from ollama import Client
from typing import List, Union

class Querier:
    def __init__(self, host: str, embed_model: str):
        self.client = Client(host)
        self.model = embed_model

    def get_embedding(self, data: Union[str, List[str]]) -> List[List[float]]:
        # suggest that the data is just a few strings or one string
        response = self.client.embed(model=self.model, input=data)
        return response['embeddings']

    def find_similar_chunks(
        self,
        query_embedding: List[List[float]],
        chunk_embedding: List[List[float]],
        top_k: int = 10
    ):  
        chunk_embedding = np.array(chunk_embedding, dtype=np.float32)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        similarities: np.ndarray = np.dot(chunk_embedding, query_embedding.T)
        return similarities.argsort(axis=0)[-top_k:][::-1].reshape(-1)
    
if __name__ == '__main__':
    querier = Querier(
        host='http://localhost:11434', 
        embed_model='all-minilm'
    )
    chunks = [
        'Apple is a fruit.',
        'Cat is an animal.',
        'Banana is a fruit.',
        'Carrot is a vegetable.',
        'Dog is an animal.',
        'Elephant is an animal.'
    ]
    queries = [
        # 'Fruit is delicious.',
        'Peach'
    ]
    query_embedding = querier.get_embedding(queries)
    chunk_embedding = querier.get_embedding(chunks)
    top_k = 6
    similar_indices = querier.find_similar_chunks(query_embedding, chunk_embedding, top_k)
    print(f"Top {top_k} similar chunks to '{queries[0]}':")
    for idx in similar_indices:
        print(f"ID: {idx}, Chunk: {chunks[idx]}")

# python -m src.query