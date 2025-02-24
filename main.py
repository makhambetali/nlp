from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import numpy as np
import os

app = FastAPI(title="Medical Services Search API", version="1.0")

MODEL_NAME = "cointegrated/LaBSE-en-ru"
embedder = SentenceTransformer(MODEL_NAME)

scopes = [
    "Доктор проводит медицинский осмотр пациента.",
    "Кардиолог лечит заболевания сердца и сосудов.",
    "Педиатр занимается диагностикой и лечением детей.",
    "Невролог проводит обследование нервной системы.",
    "Гастроэнтеролог помогает при заболеваниях желудка и кишечника.",
    "Офтальмолог проверяет зрение и подбирает очки.",
    "Стоматолог удаляет зубы и лечит кариес.",
    "Рентгенолог делает снимки органов и костей.",
    "Медсестра делает прививки и ставит капельницы.",
    "Диетолог разрабатывает индивидуальный план питания.",
    "Физиотерапевт проводит массаж и лечебную гимнастику.",
    "Акушер-гинеколог ведет беременность и принимает роды.",
    "Дерматолог лечит кожные заболевания, акне и экзему.",
    "Эндокринолог занимается лечением диабета и гормональных нарушений.",
    "Отоларинголог лечит заболевания ушей, горла и носа.",
    "Онколог диагностирует и лечит рак.",
    "Психиатр консультирует пациентов с психическими расстройствами.",
    "Аллерголог выявляет и лечит аллергические реакции.",
    "Уролог занимается лечением заболеваний мочевыводящих путей.",
    "Ревматолог помогает пациентам с заболеваниями суставов и костей.",
]

INDEX_FILE = "medical_annoy_index.ann"
VECTOR_DIMENSION = 768
METRIC = "angular"

annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)

if os.path.exists(INDEX_FILE):
    annoy_index.load(INDEX_FILE)
else:
    corpus_embeddings = embedder.encode(scopes, convert_to_tensor=False)
    for i, vector in enumerate(corpus_embeddings):
        annoy_index.add_item(i, vector)
    annoy_index.build(100)
    annoy_index.save(INDEX_FILE)

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search/")
async def search(query_data: SearchQuery):
    query_embedding = embedder.encode(query_data.query, convert_to_tensor=False)
    nearest_neighbors = annoy_index.get_nns_by_vector(query_embedding, query_data.top_k, include_distances=True)
    results = [
        {"service": scopes[idx], "similarity": round(1 - distance, 4)}
        for idx, distance in zip(*nearest_neighbors)
    ]
    return {"query": query_data.query, "results": results}

@app.get("/")
async def home():
    return {"message": "Medical Services Search API is running. Use /docs to test."}