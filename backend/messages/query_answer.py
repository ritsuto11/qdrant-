from pydantic import BaseModel, Field


class QeryAnswer(BaseModel):
    """LLMの回答スキーマ定義"""

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
