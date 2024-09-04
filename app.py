import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# ストレージが既に存在するか確認
if not os.path.exists("./storage"):
    # ドキュメントを読み込んでインデックスを作成
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # 後で使うために保存
    index.storage_context.persist()
else:
    # 既存のインデックスを読み込む
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)

# いずれの場合もインデックスをクエリできる
query_engine = index.as_query_engine()
response = query_engine.query("りっちゃんの好きなものは?")
# response = query_engine.query("浜田家の猫の名前は?")
print(response)
