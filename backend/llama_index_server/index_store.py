import os
from typing import Optional
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core.schema import Document
from qdrant_client import QdrantClient
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.llms.gemini.base import GEMINI_MODELS
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel, Field


class IndexStore:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        model_name: str = "models/embedding-001",
        api_key: Optional[str] = os.getenv("GOOGLE_API_KEY"),
    ):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.model_name = model_name
        self.api_key = api_key

    def _create_vector_store(self) -> QdrantVectorStore:
        """QdrantVectorStoreを作成します."""
        return QdrantVectorStore(
            client=self.qdrant_client, collection_name=self.collection_name
        )

    def _create_embed_model(self) -> GeminiEmbedding:
        """GeminiEmbeddingを作成します."""
        return GeminiEmbedding(model_name=self.model_name, api_key=self.api_key)

    def build_from_vector_store(self) -> VectorStoreIndex:
        """QdrantVectorStoreからVectorStoreIndexを構築します."""
        vector_store = self._create_vector_store()
        embed_model = self._create_embed_model()
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=embed_model
        )

    def build_from_documents(self, documents: list[Document]) -> VectorStoreIndex:
        """ドキュメントからVectorStoreIndexを構築します."""
        vector_store = self._create_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        embed_model = self._create_embed_model()
        return VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            embed_model=embed_model,
        )
