"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""

import torch

from sentence_transformers import SentenceTransformer

# embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# Corpus with example sentences
# corpus = [
#     "A doctor performs a medical examination on a patient.",
#     "A cardiologist treats heart and vascular diseases.",
#     "A pediatrician diagnoses and treats children.",
#     "A neurologist examines the nervous system.",
#     "A gastroenterologist helps with stomach and intestinal diseases.",
#     "An ophthalmologist checks vision and prescribes glasses.",
#     "A dentist extracts teeth and treats cavities.",
#     "A radiologist takes X-ray images of organs and bones.",
#     "A nurse administers vaccines and IV drips.",
#     "A dietitian creates a nutrition plan for patients.",
#     "A physiotherapist provides massages and physical therapy.",
#     "An obstetrician-gynecologist monitors pregnancy and assists with childbirth.",
#     "A dermatologist treats skin diseases, acne, and eczema.",
#     "An endocrinologist manages diabetes and hormonal disorders.",
#     "An otolaryngologist treats ear, nose, and throat diseases.",
#     "An oncologist diagnoses and treats cancer.",
#     "A psychiatrist counsels patients with mental disorders.",
#     "An allergist identifies and treats allergic reactions.",
#     "A urologist treats urinary tract diseases.",
#     "A rheumatologist helps patients with joint and bone diseases.",
# ]
corpus = [
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


# Use "convert_to_tensor=True" to keep the tensors on GPU (if available)
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
# queries = [
#     "A patient complains of chest pain.",
#     "I need a doctor for my child.",
#     "Where can I get a stomach examination?",
#     "How can I check my vision and get glasses?",
#     "I need a dentist for cavity treatment.",
#     "Where can I get an X-ray scan?",
#     "I need to get vaccinated.",
#     "How do I create a proper nutrition plan?",
#     "Where can I get a back massage?",
#     "I am looking for a gynecologist for pregnancy care.",
#     "How to treat eczema and skin rashes?",
#     "Which doctor treats diabetes?",
#     "Where can I check my hearing and throat health?",
#     "I need a doctor specializing in cancer treatment.",
#     "How can I book an appointment with a psychiatrist?",
#     "Where can I get allergy treatment?",
#     "How to treat bladder problems?",
#     "What are the treatment options for arthritis?",
# ]

queries = [
    "Пациент жалуется на боль в сердце.",
    "Мне нужен врач для ребенка.",
    "Где можно пройти обследование желудка?",
    "Как проверить зрение и подобрать очки?",
    "Ищу стоматолога для лечения кариеса.",
    "Где можно сделать рентген?",
    "Мне нужно пройти вакцинацию.",
    "Как составить правильный план питания?",
    "Где можно сделать массаж для спины?",
    "Ищу гинеколога для ведения беременности.",
    "Как лечить экзему и кожные высыпания?",
    "Какой врач лечит диабет?",
    "Где можно проверить слух и горло?",
    "Мне нужен врач, который лечит рак.",
    "Как записаться к психиатру?",
    "Где лечат аллергию?",
    "Как лечить проблемы с мочевым пузырем?",
    "Какие методы лечения артрита существуют?",
]



# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]
    scores, indices = torch.topk(similarity_scores, k=top_k)

    print("\nQuery:", query)
    print("Top 5 most similar sentences in corpus:")  

    for score, idx in zip(scores, indices):
        print(corpus[idx], f"(Score: {score:.4f})")

    """
    # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
    """