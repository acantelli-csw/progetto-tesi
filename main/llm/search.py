import ast
import json
import os
import sys
import re
import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from file_embedding.db_connection import get_connection
from file_embedding.embedding import get_embedding

# Similarità coseno tra due vettori
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(prompt, top_n=10):
    prompt_embedding = get_embedding(prompt)
    prompt_embedding_json = json.dumps(prompt_embedding)

    # Passa il JSON direttamente nella query
    query = f"""
            DECLARE @prompt VECTOR(1536) = CAST('{prompt_embedding_json}' AS VECTOR(1536));

            SELECT TOP (?)
                ID,
                Content,
                NumRI,
                Progressivo,
                Titolo,
                Autore,
                Cliente,
                Embedding,
                1 - VECTOR_DISTANCE('cosine', Embedding, @prompt) AS Similarity
            FROM DocumentChunks
            ORDER BY Similarity DESC;
        """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(query, (top_n,))  # Solo top_n come parametro

    rows = cursor.fetchall()

    docs = []
    for row in rows:
        doc_id, content, numero, progressivo, titolo, autore, cliente, embedding_raw, similarity = row

        docs.append({
            "id": doc_id,
            "content": content,
            "numero": numero,
            "progressivo": progressivo,
            "titolo": titolo,
            "autore": autore,
            "cliente": cliente,
            "embedding": embedding_raw,
            "similarity": similarity
        })

    return docs

def keyword_search(prompt) -> List[Dict]:

    docs = load_documents_from_db()

    bm25_index, doc_ids = build_bm25_index(docs)

    top_docs = get_top_documents(prompt, bm25_index, doc_ids, docs, top_k=3)
    
    if not top_docs:
        print("Nessun risultato trovato\n")

    return top_docs

def load_documents_from_db() -> List[Dict]:
    
    # Leggi tutti i documenti dal DB
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT ID, Content, NumRI, Progressivo, Titolo, Autore, Cliente, Embedding
    FROM DocumentChunks
    """)
    rows = cursor.fetchall()
    
    docs = []
    for row in rows:
        doc_id, content, numero, progressivo, titolo, autore, cliente, embedding_raw = row
        
        # Parse embedding se necessario
        if isinstance(embedding_raw, str):
            embedding = np.array(ast.literal_eval(embedding_raw), dtype=float)
        else:
            embedding = embedding_raw
        
        docs.append({
            "id": doc_id,
            "content": content,
            "numero": numero,
            "progressivo": progressivo,
            "titolo": titolo,
            "autore": autore,
            "cliente": cliente,
            "embedding": embedding
        })
    
    return docs

def get_top_documents(
    query: str,
    bm25_index: BM25Okapi,
    doc_ids: List[str],
    docs: List[Dict],
    top_k: int = 10
) -> List[Dict]:

    results = search_bm25(query, bm25_index, doc_ids, top_k)
    
    # Crea mappa doc_id -> documento per accesso veloce
    doc_map = {str(doc["id"]): doc for doc in docs}
    
    top_docs = []
    for doc_id, score in results:
        doc = doc_map[doc_id].copy()
        doc["bm25_score"] = score
        top_docs.append(doc)
    
    return top_docs

def search_bm25(
    query: str,
    bm25_index: BM25Okapi,
    doc_ids: List[str],
    top_k: int = 10
) -> List[Tuple[str, float]]:

    # Tokenizza la query
    query_tokens = tokenize(query)
    
    # Ottieni gli scores BM25
    scores = bm25_index.get_scores(query_tokens)
    
    # Crea lista di (doc_id, score)
    doc_scores = list(zip(doc_ids, scores))
    
    # Ordina per score decrescente e filtra score > 0
    ranked_docs = sorted(
        [(doc_id, score) for doc_id, score in doc_scores if score > 0],
        key=lambda x: x[1],
        reverse=True
    )
    
    return ranked_docs[:top_k]

def build_bm25_index(docs: List[Dict]) -> Tuple[BM25Okapi, List[str]]:

    doc_ids = [str(doc["id"]) for doc in docs]
    
    # Tokenizza tutti i documenti rimuovendo stopwords italiane
    tokenized_docs = [tokenize(doc["content"]) for doc in docs]
    
    # Crea l'indice BM25
    bm25_index = BM25Okapi(tokenized_docs)
    
    return bm25_index, doc_ids

def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    # Rimuovi stopwords
    tokens = [t for t in tokens if t not in ITALIAN_STOPWORDS]
    return tokens

ITALIAN_STOPWORDS = {
    'a', 'adesso', 'ai', 'al', 'alla', 'allo', 'allora', 'altre', 'altri', 'altro',
    'anche', 'ancora', 'avere', 'aveva', 'avevano', 'ben', 'buono', 'che', 'chi',
    'cinque', 'comprare', 'con', 'consecutivo', 'consecutivi', 'cosa', 'cui', 'da',
    'del', 'della', 'dello', 'dentro', 'deve', 'devo', 'di', 'doppio', 'due', 'e',
    'ecco', 'fare', 'fine', 'fino', 'fra', 'gente', 'giu', 'ha', 'hai', 'hanno', 'ho',
    'il', 'indietro', 'invece', 'io', 'la', 'lavoro', 'le', 'lei', 'lo', 'loro', 'lui',
    'lungo', 'ma', 'me', 'meglio', 'molta', 'molti', 'molto', 'nei', 'nella', 'no',
    'noi', 'nome', 'nostro', 'nove', 'nuovi', 'nuovo', 'o', 'oltre', 'ora', 'otto',
    'peggio', 'pero', 'persone', 'piu', 'poco', 'primo', 'promesso', 'qua', 'quarto',
    'quasi', 'quattro', 'quello', 'questo', 'qui', 'quindi', 'quinto', 'rispetto',
    'sara', 'secondo', 'sei', 'sembra', 'sembrava', 'senza', 'sette', 'sia', 'siamo',
    'siete', 'solo', 'sono', 'sopra', 'soprattutto', 'sotto', 'stati', 'stato', 'stesso',
    'su', 'subito', 'sul', 'sulla', 'tanto', 'te', 'tempo', 'terzo', 'tra', 'tre',
    'triplo', 'ultimo', 'un', 'una', 'uno', 'va', 'vai', 'voi', 'volte', 'vostro'
}

def semantic_search_old(prompt):

    docs = load_documents_from_db()

    prompt_embedding = get_embedding(prompt)

    for doc in docs:
        doc["similarity"] = cosine_similarity(prompt_embedding, doc["embedding"])

    # Ordina decrescente per similarità
    docs_sorted = sorted(docs, key=lambda x: x["similarity"], reverse=True)

    # Prendi i top N documenti più simili
    top_n = 10
    top_docs = docs_sorted[:top_n]
        
    return top_docs