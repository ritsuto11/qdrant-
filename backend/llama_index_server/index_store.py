import os
from typing import Optional
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.base.embeddings.base import (
    BaseEmbedding,
)
from llama_index.core.schema import Document
from qdrant_client import QdrantClient
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore


class IndexStore:
    def __init__(
        self,
        qdrant_client: QdrantClient = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=os.getenv("QDRANT_PORT", 6333),
        ),
        model_name: str = "models/embedding-001",
        api_key: Optional[str] = os.getenv("GOOGLE_API_KEY"),
    ):
        """
        IndexStoreを初期化します.

        Args:
            qdrant_client (QdrantClient): QdrantClient. デフォルトはローカルのQdrantClient
            model_name (str): Embeddingモデル名. デフォルトは"models/embedding-001"
            api_key (Optional[str]): Google APIキー. デフォルトは環境変数"GOOGLE_API_KEY"
        """

        self.qdrant_client = qdrant_client
        self.model_name = model_name
        self.api_key = api_key

    def _create_qdrant_vector_store(
        self, collection_name: str
    ) -> BasePydanticVectorStore:
        """QdrantVectorStoreを作成します."""

        return QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
        )

    def _create_gemini_embed_model(self) -> BaseEmbedding:
        """GeminiEmbeddingを作成します."""

        return GeminiEmbedding(model_name=self.model_name, api_key=self.api_key)

    def build_from_vector_store(self, collection_name: str) -> VectorStoreIndex:
        """
        QdrantVectorStoreからVectorStoreIndexを構築します.

        Args:
            collection_name (str): ベクトルを格納するコレクション名
        """

        vector_store = self._create_qdrant_vector_store(collection_name)
        embed_model = self._create_gemini_embed_model()
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=embed_model
        )

    def build_from_documents(
        self, documents: list[Document], collection_name: str
    ) -> VectorStoreIndex:
        """
        ドキュメントからVectorStoreIndexを構築します.

        Args:
            documents (list[Document]): ドキュメントオブジェクトのリスト
            collection_name (str): ドキュメントのべクトルを格納するコレクション名
        """

        vector_store = self._create_qdrant_vector_store(collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        embed_model = self._create_gemini_embed_model()
        return VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            embed_model=embed_model,
        )
