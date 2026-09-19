import os
import httpx
from typing import Dict, Any, List, Optional
from src.config.settings import settings

def load_secret(env_file_var: str, fallback_secret: str) -> str:
    """Read credentials from mounted container secrets (/run/secrets/*) or fallback to env."""
    secret_path = os.getenv(env_file_var)
    if secret_path and os.path.exists(secret_path):
        with open(secret_path, "r") as f:
            return f.read().strip()
    return fallback_secret

class NVIDIA_NIM_Client:
    """Async adapter interface to interact with authenticated NVIDIA NIM microservices."""
    
    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = endpoint or settings.nim_llm_endpoint
        api_key = load_secret("NGC_API_KEY_FILE", settings.ngc_api_key.get_secret_value())
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
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
