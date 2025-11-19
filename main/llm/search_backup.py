import json
import os
import sys
import re
from datetime import datetime
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from file_embedding.db_connection import get_connection
from file_embedding.embedding import get_embedding

def semantic_search(prompt, top_n=10):

    prompt_embedding = get_embedding(prompt)
    prompt_embedding_json = json.dumps(prompt_embedding)

    # Query per calcolo similarità coseno direttamente nel DB
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

    cursor.execute(query, (top_n,))

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
    bm25_index, doc_ids = get_bm25_index()
    docs = load_documents_for_bm25()
    top_docs = get_top_documents(prompt, bm25_index, doc_ids, docs, top_k=3)
    
    if not top_docs:
        print("Nessun risultato trovato dalla keyword search\n")
    return top_docs


def get_bm25_index():
    global _bm25_cache
    # Controlla se l'indice è già presente
    if _bm25_cache["index"] is not None:
        return _bm25_cache["index"], _bm25_cache["doc_ids"]
    # Altrimenti lo crea
    docs = load_documents_for_bm25()
    bm25_index, doc_ids = build_bm25_index(docs)
    
    _bm25_cache["index"] = bm25_index
    _bm25_cache["doc_ids"] = doc_ids
    _bm25_cache["last_update"] = datetime.now()
    return bm25_index, doc_ids


def load_documents_for_bm25() -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT ID, Content, NumRI, Progressivo, Titolo, Autore, Cliente
    FROM DocumentChunks
    """)
    
    rows = cursor.fetchall()
    return [{"id": row[0], "content": row[1], "numero": row[2], 
             "progressivo": row[3], "titolo": row[4], "autore": row[5], 
             "cliente": row[6]} for row in rows]


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


def get_top_documents(query: str, bm25_index: BM25Okapi, doc_ids: List[str], docs: List[Dict], top_k: int = 10) -> List[Dict]:

    results = search_bm25(query, bm25_index, doc_ids, top_k)

    # Crea mappa doc_id -> documento per accesso diretto
    doc_map = {str(doc["id"]): doc for doc in docs}
    
    top_docs = []
    for doc_id, score in results:
        doc = doc_map[doc_id].copy()
        doc["bm25_score"] = score
        top_docs.append(doc)
    return top_docs


def search_bm25(query: str, bm25_index: BM25Okapi, doc_ids: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
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


# Variabile globale cache indice BM25
_bm25_cache = {
    "index": None,
    "doc_ids": None,
    "last_update": None
}

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