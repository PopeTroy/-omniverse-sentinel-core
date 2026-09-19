import asyncio
from typing import List, Dict, Any
from src.nim.service_client import NVIDIA_NIM_Client

class EphemeralRAGPipeline:
    """A transient retrieval pipeline that instantiates on watchdog triggers

    and dissolves immediately after state ingestion.
    """
    def __init__(self, nim_client: NVIDIA_NIM_Client):
        self.nim_client = nim_client
        self.vector_cache: Dict[str, List[float]] = {}

    async def spawn_and_retrieve(self, query_text: str, database_records: List[str]) -> str:
        # Build ephemeral embedding context
        query_vector = await self.nim_client.generate_embeddings(query_text)
        
        # Fast in-memory similarity matching (Cosine)
        best_match = ""
        highest_score = -1.0
        
        for record in database_records:
            doc_vector = await self.nim_client.generate_embeddings(record)
            score = self._cosine_similarity(query_vector, doc_vector)
            if score > highest_score:
                highest_score = score
                best_match = record
                
        return best_match

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        import numpy as np
        a, b = np.array(vec_a), np.array(vec_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    async def dissolve(self):
        """Purge all ephemeral memory references."""
        self.vector_cache.clear()
