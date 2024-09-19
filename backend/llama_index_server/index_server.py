from llama_index.llms.gemini import Gemini
from llama_index.llms.gemini.base import GEMINI_MODELS
from llama_index_server.index_store import IndexStore
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.llms.llm import LLM
from llama_index.core.base.base_query_engine import BaseQueryEngine
from messages.query_answer import QeryAnswer


def create_query_engine_from_documents(
    llm: LLM = Gemini(models=GEMINI_MODELS[6]),
    collection_name: str = "collection",
    input_dir: str = "./documents/",
) -> BaseQueryEngine:
    """
    ドキュメントからクエリエンジンを作成する.

    Args:
        llm (LLM): LLMモデル. デフォルトはGEMINIモデル
        collection_name (str): コレクション名. デフォルトは"collection"
        input_dir (str): ドキュメントのディレクトリパス. デフォルトは"./documents/"
    """

    index_builder = IndexStore()
    index = index_builder.build_from_documents(
        documents=SimpleDirectoryReader(input_dir).load_data(),
        collection_name=collection_name,
    )
    return index.as_query_engine(
        llm=llm,
        output_cls=QeryAnswer,
    )


def create_query_engine_from_vector_store(
    llm: LLM = Gemini(models=GEMINI_MODELS[6]),
    collection_name: str = "collection",
) -> BaseQueryEngine:
    """
    ベクトルストアからクエリエンジンを作成する.

    Args:
        llm (LLM): LLMモデル. デフォルトはGEMINIモデル
        collection_name (str): コレクション名. デフォルトは"collection"
    """

    index_builder = IndexStore()
    index = index_builder.build_from_vector_store(collection_name)
    return index.as_query_engine(
        llm=Gemini(models=llm),
        output_cls=QeryAnswer,
    )


if __name__ == "__main__":
    """
    このファイルを実行する場合は、以下のコマンドで実行

    ```sh
    PYTHONPATH=$PYTHONPATH:/app python llama_index_server/index_server.py
    ```

    """
    query_engine = create_query_engine_from_documents()
    response = query_engine.query("さつきの家族構成は？")
    print(f"{response}")
