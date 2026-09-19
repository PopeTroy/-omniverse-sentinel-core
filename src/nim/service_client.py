import httpx
from typing import Dict, Any, Optional

class NVIDIA_NIM_Client:
    """Async adapter interface to interact with hosted or local NVIDIA NIM microservices."""
    
    def __init__(self, endpoint: str, api_key: Optional[str] = None):
        self.endpoint = endpoint
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else ""
        }

    async def generate_reasoning(self, prompt: str, context: str) -> Dict[str, Any]:
        payload = {
            "model": "nemotron-4-340b-instruct",
            "messages": [
                {"role": "system", "content": f"Context: {context}"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1024
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.endpoint}/chat/completions",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()

    async def generate_embeddings(self, text: str) -> List[float]:
        payload = {"input": text, "model": "nv-embedqa-e5-v5"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{self.endpoint}/embeddings",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
