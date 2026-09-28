import os


class Config:
    def __init__(self) -> None:
        chroma_api_key = os.getenv("CHROMA_API_KEY")
        if not chroma_api_key:
            raise ValueError("Missing required environment variable: 'CHROMA_API_KEY'")

        chroma_tenant = os.getenv("CHROMA_TENANT")
        if not chroma_tenant:
            raise ValueError("Missing required environment variable: 'CHROMA_TENANT'")

        chroma_database = os.getenv("CHROMA_DATABASE")
        if not chroma_database:
            raise ValueError("Missing required environment variable: 'CHROMA_DATABASE'")

        self._chroma_api_key = chroma_api_key
        self._chroma_tenant = chroma_tenant
        self._chroma_database = chroma_database

    @property
    def chroma_api_key(self) -> str:
        return self._chroma_api_key

    @property
    def chroma_tenant(self) -> str:
        return self._chroma_tenant

    @property
    def chroma_database(self) -> str:
        return self._chroma_database
