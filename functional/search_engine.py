import os
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from config import config

MODEL_NAME = config.MODEL_NAME
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

VECTOR_DIMENSION = embedder.get_sentence_embedding_dimension()

annoy_index = AnnoyIndex(VECTOR_DIMENSION, "angular")

INDEX_FILE = config.INDEX_FILE

def load_or_build_index():
    if os.path.exists(INDEX_FILE):
        annoy_index.load(INDEX_FILE)
    else:
        corpus_embeddings = embedder.encode(scopes, convert_to_tensor=False)
        for i, vector in enumerate(corpus_embeddings):
            annoy_index.add_item(i, vector)
        annoy_index.build(100)
        annoy_index.save(INDEX_FILE)


load_or_build_index()

def search_services(query, top_k=5):
    query_embedding = embedder.encode(query, convert_to_tensor=False)
    nearest_neighbors = annoy_index.get_nns_by_vector(query_embedding, top_k, include_distances=True)
    results = [
        {"service": scopes[idx], "similarity": round(1 - distance, 4)}
        for idx, distance in zip(*nearest_neighbors)
    ]
    return results
