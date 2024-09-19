from fastapi import FastAPI
from llama_index_server import (
    create_query_engine_from_documents,
    create_query_engine_from_vector_store,
)
from messages import QeryAnswer, QueryItem

app = FastAPI()
# query_engine = create_query_engine_from_documents()
query_engine = create_query_engine_from_vector_store()


@app.post("/query/", response_model=QeryAnswer)
async def query(item: QueryItem):
    res = query_engine.query(item.query)
    return res


@app.post("/create_collection/")
async def create_collection():
    query_engine = create_query_engine_from_documents()
    return {"status": "success"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
