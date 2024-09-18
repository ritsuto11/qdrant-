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


class Documents:
    """ドキュメントを表す値オブジェクト."""

    def __init__(self, document: Document):
        """初期化."""
        self._document = document

    @property
    def value(self) -> list[Document]:
        """ドキュメントを取得."""
        return self._document

    @classmethod
    def from_directory(cls, file_path: str):
        """ファイルパスからドキュメントを読み込みます."""
        return cls(SimpleDirectoryReader(file_path).load_data())


class VectorStoreIndexBuilder:
    """VectorStoreIndexを構築するためのクラス."""

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


if __name__ == "__main__":

    # Qdrantクライアントの初期化
    qdrant_client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=os.getenv("QDRANT_PORT", 6333),
    )

    COLLECTION_NAME = "totoro"  # Qdrantのコレクション名
    index_builder = VectorStoreIndexBuilder(
        qdrant_client=qdrant_client, collection_name=COLLECTION_NAME
    )
    # index = index_builder.build_from_documents(
    #     Documents.from_directory("./data/totoro/").value
    # )
    # # または
    index = index_builder.build_from_vector_store()

    class Response(BaseModel):
        answer: str = Field(description="質問に対する回答")
        reason: str = Field(description="回答の理由", default="")
        summary: str = Field(description="回答の要約", default="")
        source_file_name: str = Field(
            description="回答のソースに使った ファイル名", default=""
        )
        source_file_path: str = Field(
            description="回答のソースに使った ファイルパス", default=""
        )
        related_topics: list[str] = Field(description="関連するトピック", default=[])

    query_engine = index.as_query_engine(
        llm=Gemini(models=GEMINI_MODELS[6]),
        output_cls=Response,
    )

    # クエリの実行
    response = query_engine.query("さつきの家族構成は？")
    print(f"{response}")
