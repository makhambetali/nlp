from fastapi import FastAPI
from routers import search_routes

app = FastAPI(title="Medical Services Search API", version="1.0")

app.include_router(search_routes.router)

@app.get("/")
async def home():
    return {"message": "Medical Services Search API is running. Use /docs to test."}
