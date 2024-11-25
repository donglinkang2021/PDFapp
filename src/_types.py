from dataclasses import dataclass
from typing import List

@dataclass
class EmbeddingResponse:
    model: str
    embeddings: List[List[float]]
    total_duration: int = None
    load_duration: int = None
    prompt_eval_count: int = None

    @staticmethod
    def from_dict(data: dict) -> 'EmbeddingResponse':
        return EmbeddingResponse(
            model=data['model'],
            embeddings=data['embeddings'],
            total_duration=data.get('total_duration'),
            load_duration=data.get('load_duration'),
            prompt_eval_count=data.get('prompt_eval_count')
        )
