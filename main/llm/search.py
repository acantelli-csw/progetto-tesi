import ast
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from file_embedding.db_connection import get_connection
from file_embedding.embedding import get_embedding

# TODO sposta similarity search sul DB SQL Server 25 - NON SONO RIUSCITO
# TODO implement keywords search

# Similarità coseno tra due vettori.
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(prompt):

    prompt_embedding = get_embedding(prompt)

    # Leggi tutti gli embedding dal DB
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT Embedding, Content, NumRI, Progressivo, TitoloRI, Autore, Cliente 
    FROM DocumentChunks WHERE Embedding IS NOT NULL
    """)
    rows = cursor.fetchall()

    docs = []
    for row in rows:
        embedding_raw, content, numero, progressivo, titolo, autore, cliente = row
        embedding = np.array(ast.literal_eval(embedding_raw), dtype=float)
        docs.append({"embedding": embedding, "content": content, "numero": numero, 
                    "progressivo": progressivo,"titolo": titolo,  "autore": autore, "cliente": cliente,})
        
    for doc in docs:
        doc["similarity"] = cosine_similarity(prompt_embedding, doc["embedding"])

    # Ordina decrescente per similarità
    docs_sorted = sorted(docs, key=lambda x: x["similarity"], reverse=True)

    # Prendi i top N documenti più simili
    top_n = 10
    top_docs = docs_sorted[:top_n]

    output_lines = []
    for d in top_docs:
        line = f"{d['numero']} - Chunk n°{d['progressivo']} -> Similarità: {d['similarity']}"
        output_lines.append(line)
        
    return top_docs, output_lines