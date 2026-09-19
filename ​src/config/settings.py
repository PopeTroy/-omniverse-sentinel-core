import os
from pydantic import BaseSettings, Field, SecretStr

class SystemSettings(BaseSettings):
    # NVIDIA NGC & API Credentials
    ngc_api_key: SecretStr = Field(..., env="NGC_API_KEY")
    nim_llm_endpoint: str = Field("http://localhost:8000/v1", env="NIM_LLM_ENDPOINT")
    nim_embed_endpoint: str = Field("http://localhost:8003/v1", env="NIM_EMBED_ENDPOINT")
    nim_rerank_endpoint: str = Field("http://localhost:8004/v1", env="NIM_RERANK_ENDPOINT")

    # Vector DB Secrets & Config
    milvus_host: str = Field("localhost", env="MILVUS_HOST")
    milvus_port: int = Field(19530, env="MILVUS_PORT")
    milvus_token: SecretStr = Field(SecretStr(""), env="MILVUS_TOKEN")

    # Runtime Hardware & Parallelism Configuration
    trt_max_workspace_mb: int = Field(2048, env="TRT_MAX_WORKSPACE_MB")
    trt_fp16_enable: bool = Field(True, env="TRT_FP16_ENABLE")
    nim_cache_path: str = Field("/opt/nim/.cache", env="NIM_CACHE_PATH")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global Immutable Instance
settings = SystemSettings()
