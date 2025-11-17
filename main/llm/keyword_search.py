import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from file_embedding.db_connection import get_connection
import sqlite3
from typing import List, Dict

# SETUP INIZIALE DA ESEGUIRE SOLO UNA VOLTA per creare la tabella FTS
def setup_fts_table(conn):
    cursor = conn.cursor()
    
    conn.autocommit = True
    # Crea catalogo full-text se non esiste
    cursor.execute("""
    IF NOT EXISTS (SELECT * FROM sys.fulltext_catalogs WHERE name = 'ftCatalog')
    BEGIN
        CREATE FULLTEXT CATALOG ftCatalog AS DEFAULT;
    END
    """)
    
    # Crea indice full-text su più colonne se non esiste
    cursor.execute("""
    IF NOT EXISTS (
        SELECT * 
        FROM sys.fulltext_indexes fi
        JOIN sys.tables t ON fi.object_id = t.object_id
        WHERE t.name = 'DocumentChunks'
    )
    BEGIN
        CREATE FULLTEXT INDEX ON DocumentChunks
        (
            Content LANGUAGE 1040,
            Numero LANGUAGE 1040,
            Titolo LANGUAGE 1040,
            Autore LANGUAGE 1040,
            Cliente LANGUAGE 1040
        )
        KEY INDEX PK_DocumentChunks
        WITH STOPLIST = SYSTEM;
    END
    """)
    
    # Popola inizialmente l’indice full-text
    cursor.execute("ALTER FULLTEXT INDEX ON DocumentChunks START FULL POPULATION;")
    conn.autocommit = False

    print("✓ Tabella e indice full-text multi-colonna creati con successo")



# ============================================================
# FUNZIONI DI RICERCA OTTIMIZZATE
# ============================================================

def keyword_search_optimized(prompt: str, top_k: int = 3) -> List[Dict]:
    """
    Esegue keyword search usando l'indice FTS5 del database.
    
    Performance:
    - O(log n) invece di O(n)
    - Nessun caricamento in memoria
    - Ranking BM25 nativo
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Preprocessa la query (rimuovi caratteri speciali che potrebbero causare errori FTS)
    cleaned_query = preprocess_fts_query(prompt)
    
    if not cleaned_query:
        return []
    
    # Query FTS5 con ranking BM25 nativo
    # rank: score negativo (più è basso, migliore è il match)
    cursor.execute("""
    SELECT 
        dc.ID,
        dc.Content,
        dc.NumRI,
        dc.Progressivo,
        dc.Titolo,
        dc.Autore,
        dc.Cliente,
        -rank as bm25_score
    FROM DocumentChunks_FTS fts
    JOIN DocumentChunks dc ON fts.rowid = dc.ID
    WHERE DocumentChunks_FTS MATCH ?
    ORDER BY rank
    LIMIT ?
    """, (cleaned_query, top_k))
    
    rows = cursor.fetchall()
    
    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "content": row[1],
            "numero": row[2],
            "progressivo": row[3],
            "titolo": row[4],
            "autore": row[5],
            "cliente": row[6],
            "bm25_score": row[7]
        })
    
    return results


def preprocess_fts_query(query: str) -> str:
    """
    Preprocessa la query per FTS5.
    FTS5 usa una sintassi speciale: AND è implicito, OR e NOT richiedono sintassi specifica.
    """
    # Rimuovi caratteri che possono causare errori di sintassi FTS
    query = query.replace('"', '').replace("'", "")
    
    # Rimuovi stopwords italiane comuni (opzionale)
    stopwords = {
        'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una',
        'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra',
        'e', 'o', 'ma', 'anche', 'che', 'non'
    }
    
    words = query.lower().split()
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    
    if not filtered_words:
        return query.lower()  # Fallback alla query originale
    
    # Unisci con AND implicito (spazio)
    return ' '.join(filtered_words)


def keyword_search_advanced(
    prompt: str, 
    top_k: int = 3,
    search_in_title: bool = True,
    boost_title_matches: bool = True
) -> List[Dict]:
    """
    Ricerca avanzata con opzioni aggiuntive.
    
    Args:
        prompt: Query di ricerca
        top_k: Numero di risultati da restituire
        search_in_title: Se True, cerca anche nei titoli
        boost_title_matches: Se True, dà priorità ai match nel titolo
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cleaned_query = preprocess_fts_query(prompt)
    
    if not cleaned_query:
        return []
    
    if boost_title_matches:
        # Cerca prima nei titoli, poi nel contenuto
        cursor.execute("""
        WITH title_matches AS (
            SELECT 
                dc.ID,
                dc.Content,
                dc.NumRI,
                dc.Progressivo,
                dc.Titolo,
                dc.Autore,
                dc.Cliente,
                -rank * 2 as bm25_score
            FROM DocumentChunks_FTS fts
            JOIN DocumentChunks dc ON fts.rowid = dc.ID
            WHERE DocumentChunks_FTS MATCH 'Titolo: ' || ?
        ),
        content_matches AS (
            SELECT 
                dc.ID,
                dc.Content,
                dc.NumRI,
                dc.Progressivo,
                dc.Titolo,
                dc.Autore,
                dc.Cliente,
                -rank as bm25_score
            FROM DocumentChunks_FTS fts
            JOIN DocumentChunks dc ON fts.rowid = dc.ID
            WHERE DocumentChunks_FTS MATCH 'Content: ' || ?
        )
        SELECT * FROM (
            SELECT * FROM title_matches
            UNION
            SELECT * FROM content_matches
        )
        ORDER BY bm25_score DESC
        LIMIT ?
        """, (cleaned_query, cleaned_query, top_k))
    else:
        cursor.execute("""
        SELECT 
            dc.ID,
            dc.Content,
            dc.NumRI,
            dc.Progressivo,
            dc.Titolo,
            dc.Autore,
            dc.Cliente,
            -rank as bm25_score
        FROM DocumentChunks_FTS fts
        JOIN DocumentChunks dc ON fts.rowid = dc.ID
        WHERE DocumentChunks_FTS MATCH ?
        ORDER BY rank
        LIMIT ?
        """, (cleaned_query, top_k))
    
    rows = cursor.fetchall()
    
    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "content": row[1],
            "numero": row[2],
            "progressivo": row[3],
            "titolo": row[4],
            "autore": row[5],
            "cliente": row[6],
            "bm25_score": row[7]
        })
    
    return results


# ============================================================
# FUNZIONI DI UTILITÀ
# ============================================================

def rebuild_fts_index(conn: sqlite3.Connection):
    """Ricostruisce l'indice FTS5 (utile dopo modifiche massive al DB)"""
    cursor = conn.cursor()
    cursor.execute("INSERT INTO DocumentChunks_FTS(DocumentChunks_FTS) VALUES('rebuild')")
    conn.commit()
    print("✓ Indice FTS5 ricostruito")


def optimize_fts_index(conn: sqlite3.Connection):
    """Ottimizza l'indice FTS5 per migliorare le performance"""
    cursor = conn.cursor()
    cursor.execute("INSERT INTO DocumentChunks_FTS(DocumentChunks_FTS) VALUES('optimize')")
    conn.commit()
    print("✓ Indice FTS5 ottimizzato")


# ============================================================
# ESEMPIO DI UTILIZZO
# ============================================================

if __name__ == "__main__":
    # Setup iniziale (eseguire UNA VOLTA)
    # conn = get_connection()
    # setup_fts_table(conn)
    
    # Utilizzo normale
    results = keyword_search_optimized("contratto di appalto", top_k=5)
    
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc['titolo']} (Score: {doc['bm25_score']:.2f})")
        print(f"   Cliente: {doc['cliente']}")
        print(f"   Contenuto: {doc['content'][:150]}...")