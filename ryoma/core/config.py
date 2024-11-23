from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Ryoma"
    PROJECT_DESCRIPTION: str = (
        "An intelligent agent project based on various LLM services"
    )
    VERSION: str = "0.1.0"

    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()