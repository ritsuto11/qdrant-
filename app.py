import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core.schema import Document
from pydantic import BaseModel, Field
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.llms.gemini.base import GEMINI_MODELS

import os


class Response(BaseModel):
    answer: str = Field(description="質問に対する回答")
    summary: str = Field(description="回答の要約", default="")
    source: str = Field(description="回答のソース", default="")
    related_topics: list[str] = Field(description="関連するトピック", default=[])


def load_docs_from_file(file_path: str) -> list[Document]:
    documents = SimpleDirectoryReader(file_path).load_data()
    return documents


import os


def get_vector_store_index(
    qdrant_client: QdrantClient,
    collection_name: str,
    model_name: str = "models/embedding-001",
    api_key: str = os.getenv("GOOGLE_API_KEY"),
) -> VectorStoreIndex:
    """
    QdrantVectorStoreとGeminiEmbeddingを用いたVectorStoreIndexを初期化して返します。

    Args:
        qdrant_client: Qdrantクライアント
        collection_name: Qdrantのコレクション名
        model_name: 埋め込みモデル名
        api_key: Google APIキー

    Returns:
        VectorStoreIndex
    """

    # QdrantVectorStoreの初期化
    vector_store = QdrantVectorStore(
        client=qdrant_client, collection_name=collection_name
    )

    # 埋め込みモデルの初期化
    embed_model = GeminiEmbedding(model_name=model_name, api_key=api_key)

    # VectorStoreIndexの初期化
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )

    return index


def get_vector_store_index_from_documents(
    documents: list[Document],
    qdrant_client: QdrantClient,
    collection_name: str,
    model_name: str = "models/embedding-001",
    api_key: str = os.getenv("GOOGLE_API_KEY"),
) -> VectorStoreIndex:
    """
    ドキュメントからQdrantVectorStoreとGeminiEmbeddingを用いたVectorStoreIndexを初期化して返します。

    Args:
        documents: ドキュメントのリスト
        qdrant_client: Qdrantクライアント
        collection_name: Qdrantのコレクション名
        model_name: 埋め込みモデル名
        api_key: Google APIキー

    Returns:
        VectorStoreIndex
    """

    # QdrantVectorStoreの初期化
    vector_store = QdrantVectorStore(
        client=qdrant_client, collection_name=collection_name
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 埋め込みモデルの初期化
    embed_model = GeminiEmbedding(model_name=model_name, api_key=api_key)

    # VectorStoreIndexの初期化
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    return index


if __name__ == "__main__":

    # Qdrantの設定
    QDRANT_HOST = "localhost"  # Qdrantのホスト名またはIPアドレス
    # QDRANT_HOST = "qdrant"  # Qdrantのホスト名またはIPアドレス
    QDRANT_PORT = 6333  # Qdrantのポート番号
    COLLECTION_NAME = "llama_index"  # Qdrantのコレクション名

    # Qdrantクライアントの初期化
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # index = get_vector_store_index(
    #     qdrant_client=qdrant_client,
    #     collection_name=COLLECTION_NAME,
    # )
    index = get_vector_store_index_from_documents(
        documents=SimpleDirectoryReader("./data/paul_graham/").load_data(),
        qdrant_client=qdrant_client,
        collection_name=COLLECTION_NAME,
    )

    # set Logging to DEBUG for more detailed outputs
    query_engine = index.as_query_engine(
        llm=Gemini(models=GEMINI_MODELS[6]),
        response_mode=ResponseMode.COMPACT,
        output_cls=Response,
    )
    response = query_engine.query("にゃんたの飼い主は？")
    print(f"{response}")
