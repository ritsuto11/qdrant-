import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from pydantic import BaseModel, Field
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from qdrant_client import QdrantClient


class Response(BaseModel):
    answer: str = Field(description="質問に対する回答")
    summary: str = Field(description="回答の要約", default="")
    source: str = Field(description="回答のソース", default="")
    related_topics: list[str] = Field(description="関連するトピック", default=[])


# Qdrantの設定
QDRANT_HOST = "qdrant"  # Qdrantのホスト名またはIPアドレス
QDRANT_PORT = 6333  # Qdrantのポート番号
COLLECTION_NAME = "llama_index"  # Qdrantのコレクション名

# Qdrantクライアントの初期化
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
# QdrantVectorStoreの初期化
vector_store = QdrantVectorStore(client=qdrant_client, collection_name="paul_graham")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine(
    response_mode="tree_summarize", output_cls=Response
)
response = query_engine.query("日本語で回答して。浜田家の猫の名前は？")
print(f"{response}")
