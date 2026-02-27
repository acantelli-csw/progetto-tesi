from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)
import extract_text
import db_connection
import embedding
import json
import easyocr
import os
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import bm25s
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional


class ChunkingStrategy(Enum):
    FIXED_SIZE = "fixed"
    RECURSIVE = "recursive"

@dataclass
class ChunkingConfig:

    strategy: ChunkingStrategy
    
    # Parametri per Fixed-size
    chunk_size: int
    chunk_overlap: int

    # Parametri per Recursive
    recursive_separators: Optional[List[str]] = None

    def __post_init__(self):
        """Imposta valori di default se non specificati"""
        if self.recursive_separators is None:
            # Separatori basati sulla struttura gerarchica del documento Sample_RI_Vuota.docx
            # Ordine: dal più specifico/strutturato al più generico
            standard_separators = [
                "\n## ",  # Titoli principali
                "\n**",   # Sezioni in grassetto
                "\n> ",   # Quote/blocchi indentati
                "\n\n",   # Paragrafi
                "\n",     # Righe
                ". ",     # Frasi
                ", ",     # Clausole
                " ",      # Parole
                ""
            ]
            custom_separators = [
                #"\nRichiesta Cliente\n",  
                #"\nSoluzione Proposta\n",
                "\nAnalisi Tecnica\n",       # Sezione specifica derivante dal sample RI
                "\n[OCR immagine embedded]:" # Sezione relativa all'OCR di un immagine
                "\n\n\n",           # Sezioni separate da righe vuote multiple
                "\n\n",             # Paragrafi (doppio newline)
                "\n",               # Singole righe
                ". ",               # Frasi
                ", ",               # Clausole
                " ",                # Parole
                ""                  # Caratteri (fallback)
            ]
            self.recursive_separators = custom_separators

def create_text_splitter(config: ChunkingConfig):

    if config.strategy == ChunkingStrategy.FIXED_SIZE:
        print(f"Usando Fixed-size chunking: size={config.chunk_size}, overlap={config.chunk_overlap}")
        return TokenTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    
    elif config.strategy == ChunkingStrategy.RECURSIVE:
        print(f"Usando Recursive chunking con chunk size={config.chunk_size} overlap={config.chunk_overlap}")
        #print(f"Separatori: {config.recursive_separators[:3]}...")
        return RecursiveCharacterTextSplitter(
            separators=config.recursive_separators,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
    
    else:
        raise ValueError(f"Strategia di chunking non supportata: {config.strategy}")


def process_files(config: ChunkingConfig, limit: Optional[int] = 10):

    # Inizializza reader OCR e conessione al DB
    reader = easyocr.Reader(['it', 'en'], gpu=False)
    conn = db_connection.get_connection()
    cursor = conn.cursor()
    
    # Crea il text splitter appropriato
    splitter = create_text_splitter(config)
    
    # Query per ottenere i file da processare e i relativi metadati
    limit_clause = f"top {limit}" if limit else ""
    query = f"""
    SELECT {limit_clause}
        v.InstanceID,
        MAX(CASE WHEN v.VariableName = 'NUMERO' THEN v.StringValue END) AS numero,
        MAX(CASE WHEN v.VariableName = 'CLIENTE' THEN v.StringValue END) AS cliente,
        MAX(CASE WHEN v.VariableName = 'TITOLO' THEN v.StringValue END) AS titolo,
        MAX(CASE WHEN v.VariableName = 'AUTORE' THEN v.StringValue END) AS autore,
        MAX(CASE WHEN v.VariableName = 'DOC' THEN v.StringValue END) AS documento,
        MAX(CASE WHEN v.VariableName = 'URL_DOC' THEN v.StringValue END) AS url_doc,
        MAX(f.FileData) AS FileData,
        MAX(f.Extension) AS Extension
    FROM VAR_RICSW v 
    JOIN DocumentFiles f
        ON v.InstanceID = f.InstanceID
    WHERE v.InstanceID IN (
        SELECT v2.InstanceID
        FROM VAR_RICSW v2
        WHERE v2.VariableName = 'ELABORATO'
            AND v2.BooleanValue = 0
    )
    GROUP BY v.InstanceID
    ORDER BY numero desc;
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    corpus = []
    total_files = len(rows)
    
    print(f"\n{'='*60}")
    print(f"Inizio elaborazione di {total_files} file")
    print(f"Strategia: {config.strategy.value}")
    print(f"{'='*60}\n")
    
    # Elaborazione file estratti
    for idx, row in enumerate(rows, 1):
        chunk_records = []
        instance_id, numero, cliente, titolo, autore, documento, url_doc, file_data, extension = row
        
        print(f"[{idx}/{total_files}] Elaborazione file: {numero}{extension}")
        
        # Estrazione testo + OCR immagini
        text = extract_text.extract_text_from_varbinary(file_data, extension, numero, reader)
        if not text.strip():
            print(f"  ⚠ Nessun testo estratto, file saltato\n")
            continue
        
        # Chunking del testo
        try:
            chunks = splitter.split_text(text)
            print(f"  ✓ {len(chunks)} chunk generati")
        except Exception as e:
            print(f"  ✗ Errore durante il chunking: {e}\n")
            continue
        
        # Calcolo embedding per ogni chunk
        for i, chunk in enumerate(chunks):
            emb = embedding.get_embedding(chunk)
            emb_json = json.dumps(emb)
            chunk_records.append((numero, i, cliente, titolo, autore, documento, url_doc, chunk, emb_json))
            corpus.append(chunk)
        
        # Aggiorna flag elaborazione
        cursor.execute(
            "UPDATE VAR_RICSW SET BooleanValue = 1 WHERE VariableName = 'ELABORATO' AND InstanceID = ?",
            (instance_id,)
        )
        
        # Salvataggio tutti chunk del documento nel DB in un'unica query per efficienza
        if chunk_records:
            cursor.executemany(
                "INSERT INTO DocumentChunks (NumRI, Progressivo, Cliente, Titolo, Autore, Documento, Url_doc, Content, Embedding)" \
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, CAST(CAST(? AS VARCHAR(MAX)) AS VECTOR(1536)))", 
                chunk_records
            )
        
        conn.commit()
        print(f"  ✓ Embedding salvati per file: {documento}\n")
    
    print(f"{'='*60}")
    print(f"Elaborazione completata: {total_files} file processati")
    print(f"{'='*60}\n")
    
    # Costruzione indice BM25
    build_bm25_index(conn, cursor)
    
    cursor.close()
    conn.close()


def build_bm25_index(conn, cursor):

    print(f"\n{'='*60}")
    print("Ricostruzione completa indice BM25...")
    print(f"{'='*60}\n")
    
    # Carica chunk con ID dal DB
    cursor.execute("SELECT id, Content FROM DocumentChunks ORDER BY id ASC")
    rows = cursor.fetchall()
    db_ids = [row[0] for row in rows]
    all_chunks = [row[1] for row in rows]
    
    # Tokenizzazione
    language = 'italian'
    stemmer = SnowballStemmer(language)
    stop_words = stopwords.words(language)
    print("Tokenizzazione in corso...")
    all_tokens = bm25s.tokenize(
        all_chunks,
        stopwords=stop_words,
        stemmer=lambda tokens: [stemmer.stem(t.lower()) for t in tokens]
    )
    
    # Creazione e salvataggio reverse index BM25
    print("Creazione indice BM25...")
    retriever = bm25s.BM25()
    if all_chunks:
        retriever.index(all_tokens)
    
        index_folder = "reverse_index"
        os.makedirs(index_folder, exist_ok=True)
        index_path = os.path.join(index_folder, "bm25_index")
        retriever.save(index_path)
        print(f"✓ Indice salvato in: {index_path}")

    # Aggiorna mapping nel DB
    print("Aggiornamento mapping nel database...")
    for pos, db_id in enumerate(db_ids):
        cursor.execute("""
            UPDATE DocumentChunks
            SET Bm25_index = ?
            WHERE id = ?
        """, (pos, db_id))
    
    conn.commit()
    print(f"✓ Indice BM25 completato per {len(all_chunks)} documenti\n")


# ============================================================================
# ESEMPI DI CONFIGURAZIONE
# ============================================================================

def main():
    """
    1. Fixed-size chunking (Character/Token based, dimensione chunk e overlap configurabili)
    2. Recursive chunking (basato sulla struttura gerarchica dei documenti)
    3. Semantic chunking (basato su similarità semantica)
    """
    
    # ESEMPIO 1: FIXED-SIZE
    config_fixed = ChunkingConfig(
        strategy=ChunkingStrategy.FIXED_SIZE,
        chunk_size=1024,
        chunk_overlap=150
    )
    
    # ESEMPIO 2: RECURSIVE
    config_recursive = ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=1024,
        chunk_overlap=150
    )
    

    # SCEGLI LA CONFIGURAZIONE DA USARE

    selected_config = config_recursive
    limit = 300 # Numero max di file da elaborare
    process_files(selected_config, limit)


if __name__ == "__main__":
    main()