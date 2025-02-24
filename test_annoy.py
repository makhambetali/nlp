from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import numpy as np
import os

# ✅ Создаем FastAPI-приложение
app = FastAPI(title="Medical Services Search API", version="1.0")

# ✅ Выбираем русскоязычную модель
MODEL_NAME = "cointegrated/LaBSE-en-ru"
# MODEL_NAME = "sberbank-ai/sbert_large_nlu_ru"
embedder = SentenceTransformer(MODEL_NAME)

# ✅ Определяем медицинские услуги (корпус)
scopes = [
    "Доктор проводит медицинский осмотр пациента.",
    "Кардиолог лечит заболевания сердца и сосудов.",
    "Педиатр занимается диагностикой и лечением детей.",
    "Невролог проводит обследование нервной системы.",
    "Гастроэнтеролог помогает при заболеваниях желудка и кишечника.",
    "Офтальмолог проверяет зрение и подбирает очки.",
    "Стоматолог удаляет зубы и лечит кариес.",
    "Рентгенолог делает снимки органов и костей.",
    "Медсестра делает прививки и ставит капельницы.",
    "Диетолог разрабатывает индивидуальный план питания.",
    "Физиотерапевт проводит массаж и лечебную гимнастику.",
    "Акушер-гинеколог ведет беременность и принимает роды.",
    "Дерматолог лечит кожные заболевания, акне и экзему.",
    "Эндокринолог занимается лечением диабета и гормональных нарушений.",
    "Отоларинголог лечит заболевания ушей, горла и носа.",
    "Онколог диагностирует и лечит рак.",
    "Психиатр консультирует пациентов с психическими расстройствами.",
    "Аллерголог выявляет и лечит аллергические реакции.",
    "Уролог занимается лечением заболеваний мочевыводящих путей.",
    "Ревматолог помогает пациентам с заболеваниями суставов и костей.",
]

# ✅ Путь для сохранения Annoy-индекса
INDEX_FILE = "medical_annoy_index.ann"
VECTOR_DIMENSION = 768  # Размерность эмбеддингов модели (LaBSE)
METRIC = "angular"  # Косинусное расстояние

# ✅ Создаем Annoy-индекс
annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)

# ✅ Загружаем или создаем Annoy-индекс
if os.path.exists(INDEX_FILE):
    print("🔄 Загружаем Annoy-индекс...")
    annoy_index.load(INDEX_FILE)
else:
    print("⚙️ Создаем Annoy-индекс...")
    
    # ✅ Кодируем корпус и добавляем в Annoy
    corpus_embeddings = embedder.encode(scopes, convert_to_tensor=False)

    for i, vector in enumerate(corpus_embeddings):
        annoy_index.add_item(i, vector)

    # ✅ Строим индекс (100 деревьев для баланса скорости и точности)
    annoy_index.build(100)
    annoy_index.save(INDEX_FILE)
    print("✅ Annoy-индекс сохранен!")

# ✅ Pydantic-модель для запроса
class SearchQuery(BaseModel):
    query: str
    top_k: int = 5  # Количество похожих результатов


@app.post("/search/")
async def search(query_data: SearchQuery):
    """
    Поиск ближайших медицинских услуг по смыслу.
    """
    query_embedding = embedder.encode(query_data.query, convert_to_tensor=False)

    # ✅ Ищем ближайшие `top_k` записей
    nearest_neighbors = annoy_index.get_nns_by_vector(query_embedding, query_data.top_k, include_distances=True)

    results = [
        {"service": scopes[idx], "similarity": round(1 - distance, 4)}
        for idx, distance in zip(*nearest_neighbors)
    ]

    return {"query": query_data.query, "results": results}


@app.get("/")
async def home():
    return {"message": "Medical Services Search API is running. Use /docs to test."}
