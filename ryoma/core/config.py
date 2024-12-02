from typing import Literal, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Ryoma"
    PROJECT_DESCRIPTION: str = (
        "An intelligent agent project based on various LLM services"
    )
    VERSION: str = "0.1.0"

    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    # LLM settings
    LLM_PROVIDER: Literal["aws_bedrock", "openai", "google_gemini"] = "aws_bedrock"
    LLM_MODEL_ID: str = ""

    # AWS Bedrock settings
    AWS_BEDROCK_ENDPOINT: Optional[str] = None

    # OpenAI settings
    OPENAI_API_KEY: str = ""
    OPENAI_ENDPOINT: str = ""

    GOOGLE_API_KEY: str = ""

    EMBEDDING_MODEL_NAME: str = "fasttext"

    STORAGE_VECTOR_TYPE: str = "milvus"

    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION_NAME: str = "vectors"
    MILVUS_DIM: int = 1536

    ZHIPU_API_KEY: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
