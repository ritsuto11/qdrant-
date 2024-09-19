## llamaindex × qdrant × Gemini API × 構造化出力

このリポジトリは、LlamaIndex、Qdrant、Gemini API を使用して、ドキュメントから構造化された出力を生成するREST APIのサンプルアプリケーションを提供します。


### セットアップ

1. **環境変数の設定:**
   ```bash
   cp -a env.example .env 
   ```
   `.env` ファイルで必要な環境変数を設定してください。

2. **Docker Compose で起動:**
   ```bash
   docker-compose up -d
   ```


### 使用方法

アプリケーションは、`http://localhost:8000/docs` でアクセスできる Swagger UI で提供されます。


### 技術スタック

* LlamaIndex: ドキュメントインデックスとクエリエンジン
* Qdrant: ベクトルデータベース
* Gemini API: 大規模言語モデル
* FastAPI: Web API フレームワーク
* pydantic: 構造化出力
* Docker Compose: コンテナ

