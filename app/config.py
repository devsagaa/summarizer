from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4o"

    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_size_mb: int = 50
    chunk_size: int = 400          # words per chunk
    chunk_overlap: int = 60        # word overlap between chunks
    top_k_results: int = 5         # chunks to retrieve per query
    embedding_model: str = "all-MiniLM-L6-v2"

    # Directories — relative to project root
    uploads_dir: Path = Path("uploads")
    vector_db_dir: Path = Path("vector_db")
    data_dir: Path = Path("data")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def model_post_init(self, __context) -> None:
        self.uploads_dir.mkdir(exist_ok=True)
        self.vector_db_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)


settings = Settings()
