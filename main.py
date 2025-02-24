# from fastapi import FastAPI, Query
# from annoy import AnnoyIndex
# from sentence_transformers import SentenceTransformer
# import numpy as np

# app = FastAPI()

# # Задаем размерность эмбеддингов
# VECTOR_DIMENSION = 384  # Минимальная модель "all-MiniLM-L6-v2"
# annoy_index = AnnoyIndex(VECTOR_DIMENSION, 'angular')

# # Загружаем NLP-модель для эмбеддингов
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # База данных врачей и клиник (пример)
# data = [
#     {"id": 0, "name": "Доктор Иванов, терапевт", "description": "Лечит простуду, грипп, ОРВИ, ангину, бронхит."},
#     {"id": 1, "name": "Медицинский центр 'Здоровье'", "description": "Полный спектр медицинских услуг: диагностика, терапия, вакцинация."},
#     {"id": 2, "name": "Доктор Смирнова, кардиолог", "description": "Специализируется на заболеваниях сердца, гипертонии, аритмии, инфарктах."},
#     {"id": 3, "name": "Клиника 'НейроМед'", "description": "Диагностика и лечение неврологических заболеваний, мигрени, инсульта."},
#     {"id": 4, "name": "Доктор Петров, ортопед", "description": "Лечение заболеваний костей, суставов, остеохондроза, артрита."},
#     {"id": 5, "name": "Стоматологическая клиника 'Улыбка'", "description": "Лечение зубов, удаление, установка брекетов, отбеливание."},
#     {"id": 6, "name": "Доктор Кузнецов, педиатр", "description": "Специализируется на лечении детей, вакцинации, детских инфекциях."},
#     {"id": 7, "name": "Глазная клиника 'ОптикПро'", "description": "Диагностика и лечение глазных болезней, коррекция зрения, лазерная хирургия."},
#     {"id": 8, "name": "Доктор Белова, дерматолог", "description": "Лечение кожных заболеваний, акне, экземы, псориаза, дерматита."},
#     {"id": 9, "name": "Диагностический центр 'МРТ-Плюс'", "description": "МРТ, КТ, рентген, ультразвуковая диагностика, анализы."},
#     {"id": 10, "name": "Доктор Сорокин, гастроэнтеролог", "description": "Лечение желудочно-кишечных заболеваний, гастрита, язвы, дисбактериоза."},
#     {"id": 11, "name": "Клиника 'НефроМед'", "description": "Диагностика и лечение заболеваний почек, мочекаменной болезни."},
#     {"id": 12, "name": "Доктор Орлов, эндокринолог", "description": "Специализируется на диабете, заболеваниях щитовидной железы, гормональных нарушениях."},
#     {"id": 13, "name": "Физиотерапевтический центр 'Реабилитация'", "description": "Массаж, ЛФК, восстановление после травм, физиопроцедуры."},
#     {"id": 14, "name": "Доктор Васильев, невролог", "description": "Диагностика и лечение нервных заболеваний, депрессии, бессонницы, панических атак."},
#     {"id": 15, "name": "Акушер-гинекологический центр 'Материнство'", "description": "Ведение беременности, родовспоможение, женское здоровье."},
#     {"id": 16, "name": "Доктор Кравцова, аллерголог", "description": "Диагностика и лечение аллергий, бронхиальной астмы, кожных реакций."},
#     {"id": 17, "name": "Медицинский центр 'ОстеоПлюс'", "description": "Лечение опорно-двигательного аппарата, мануальная терапия, ортопедия."},
#     {"id": 18, "name": "Доктор Федоров, онколог", "description": "Диагностика и лечение онкологических заболеваний, консультации, химиотерапия."},
#     {"id": 19, "name": "Центр психиатрии и психотерапии 'НейроБаланс'", "description": "Лечение тревожных расстройств, депрессии, психотерапия, когнитивно-поведенческая терапия."}
# ]


# # Создание эмбеддингов и добавление в Annoy
# for item in data:
#     vector = model.encode(item["description"])
#     annoy_index.add_item(item["id"], vector)

# # Строим 10 деревьев для оптимального поиска
# annoy_index.build(10)

# @app.get("/search/")
# async def search(query: str = Query(..., description="Введите запрос")):
#     query_vector = model.encode(query)
    
#     # Ищем 3 ближайших совпадения
#     nearest_neighbors = annoy_index.get_nns_by_vector(query_vector, 3, include_distances=True)
    
#     results = [{"id": i, "name": data[i]["name"],"description":data[i]["description"], "distance": d} for i, d in zip(*nearest_neighbors)]
#     return {"query": query, "results": results}


from fastapi import FastAPI, Query
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os
from sklearn.preprocessing import normalize

app = FastAPI()

# ✅ Размерность эмбеддингов (Зависит от модели!)
VECTOR_DIMENSION = 384  # "all-MiniLM-L6-v2"
METRIC = 'angular'  # Косинусное расстояние

# ✅ Загружаем NLP-модель (Выбери одну)
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # ✅ Легкая и быстрая модель

MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"  # 🔴 Сильнее, но медленнее

model = SentenceTransformer(MODEL_NAME)

# ✅ Создаем индекс Annoy
annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)

# ✅ База данных врачей и клиник
data = [
    {"id": 0, "name": "Доктор Иванов, терапевт", 
     "description": "Лечит простуду, грипп, ОРВИ, ангину, бронхит. Семейный врач, терапия, пневмония."},

    {"id": 1, "name": "Медицинский центр 'Здоровье'", 
     "description": "Полный спектр медицинских услуг: диагностика, терапия, вакцинация, чекап."},

    {"id": 2, "name": "Доктор Смирнова, кардиолог", 
     "description": "Специалист по сердечно-сосудистым заболеваниям: гипертония, аритмия, инфаркт, ишемия."},

    {"id": 3, "name": "Клиника 'НейроМед'", 
     "description": "Диагностика и лечение неврологических заболеваний: мигрени, инсульт, эпилепсия, нервы."},

    {"id": 4, "name": "Доктор Петров, ортопед", 
     "description": "Лечение суставов, артрит, ревматизм, позвоночник, остеохондроз, травмы, реабилитация."},

    {"id": 5, "name": "Стоматологическая клиника 'Улыбка'", 
     "description": "Лечение зубов, ортодонтия, удаление зубов, брекеты, импланты, протезирование."},

    {"id": 6, "name": "Доктор Кузнецов, педиатр", 
     "description": "Детский врач, лечит ОРВИ, прививки, наблюдение за новорожденными, вирусные инфекции."},

    {"id": 7, "name": "Глазная клиника 'ОптикПро'", 
     "description": "Лечение глазных болезней, коррекция зрения, катаракта, лазерная коррекция, близорукость."},

    {"id": 8, "name": "Доктор Белова, дерматолог", 
     "description": "Лечение кожи, акне, экзема, псориаз, дерматит, косметология, аллергические реакции."},

    {"id": 9, "name": "Диагностический центр 'МРТ-Плюс'", 
     "description": "МРТ, КТ, рентген, ультразвуковая диагностика, анализы, чекап."},

    {"id": 10, "name": "Доктор Орлов, эндокринолог", 
     "description": "Лечение диабета, гормональные нарушения, щитовидная железа, ожирение, метаболизм."}
]

# ✅ **Пересоздаем индекс Annoy**
EMBEDDING_FILE = "embeddings.ann"

if os.path.exists(EMBEDDING_FILE):
    print("🔄 Загружаем сохраненные эмбеддинги...")
    annoy_index.load(EMBEDDING_FILE)
else:
    print("⚙️ Генерируем новые эмбеддинги...")
    for item in data:
        vector = model.encode(item["description"])
        vector = normalize([vector])[0]  # Нормализация
        annoy_index.add_item(item["id"], vector)
        print(f"✅ Добавлен {item['name']}")

    annoy_index.build(100)  # Больше деревьев для точности
    annoy_index.save(EMBEDDING_FILE)
    print("✅ Эмбеддинги сохранены!")

# ✅ **Отладка: Проверка, есть ли данные**
print("📊 Всего объектов в Annoy:", annoy_index.get_n_items())

# ✅ **Поиск**
@app.get("/search/")
async def search(query: str = Query(..., description="Введите запрос")):
    
    query_vector = model.encode(query)
    query_vector = normalize([query_vector])[0]  # Нормализация
    

    # ✅ Выводим отладочную информацию
    print(f"\n🔍 Запрос: {query}")
    print(f"🧠 Вектор запроса: {query_vector[:5]}...")  # Выведем только первые 5 значений для проверки

    # ✅ Проверяем, что Annoy не пустой
    if annoy_index.get_n_items() == 0:
        return {"error": "Annoy Index пуст. Пересоздайте индекс."}

    # ✅ Ищем ближайшие совпадения
    nearest_neighbors = annoy_index.get_nns_by_vector(query_vector, 5, include_distances=True)

    print("🔎 Найденные ID:", nearest_neighbors[0])
    print("📏 Дистанции:", nearest_neighbors[1])

    # ✅ Фильтруем результаты (убери `if d < 0.5`, если результатов мало)
    results = [
        {"id": i, "name": data[i]["name"], "description": data[i]["description"], "distance": d}
        for i, d in zip(*nearest_neighbors)  # Расширили диапазон
    ]

    return {"query": query, "results": results}
