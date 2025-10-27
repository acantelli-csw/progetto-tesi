import numpy as np
import json
import db_connection
import embedding


# 1. Similarità coseno tra due vettori.
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# TODO sposta similarity search sul DB usando vettori nativi di SQL 25

def semantic_search(prompt):

    prompt_embedding = embedding.get_embedding(prompt)

    # Leggi tutti gli embedding dal DB
    conn = db_connection.get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT ID, Embedding FROM Files WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()
    
    docs = []
    for row in rows:
        file_id, embedding_json = row
        # TODO: switch from json.loads to vectors in SQL 25
        embedding_vector = json.loads(embedding_json)
        docs.append({"id": file_id, "embedding": embedding_vector})
        
    for doc in docs:
        doc["similarity"] = cosine_similarity(prompt_embedding, doc["embedding"])

    # Ordina decrescente per similarità
    docs_sorted = sorted(docs, key=lambda x: x["similarity"], reverse=True)

    # Prendi i top N documenti più simili
    # TODO: variare N in base al grado di similarità (soglia minima) o alla lunghezza dei documenti
    top_n = 20
    top_docs = docs_sorted[:top_n]

    for d in top_docs:
        print(d["filename"], "-> Similarità:", d["similarity"])

    return top_docs