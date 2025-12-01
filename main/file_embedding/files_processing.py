from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
import extract_text
import db_connection
import embedding
import json
import easyocr
import os
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import bm25s

# Configura il text splitter per il chunking
chunk_size = 512
chunk_overlap = 100
splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Inizializza il reader OCR una volta sola
reader = easyocr.Reader(['it', 'en'], gpu=False)

# Ottieni la connessione al DB + estrazione file senza embedding
conn = db_connection.get_connection()
cursor = conn.cursor()

# Query per ottenere i file da processare e le loro variabili associate
query = """
SELECT top 3
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

# Elaborazione file estratti
for row in rows:
    chunk_records = []
    instance_id, numero, cliente, titolo, autore, documento, url_doc, file_data, extension = row

    # Estrazione testo + OCR immagini
    text = extract_text.extract_text_from_varbinary(file_data, extension, numero, reader)
    if not text.strip():
        print(f"Nessun testo estratto dal file {numero}{extension}, salto il file.")
        continue

    chunks = splitter.split_text(text)

    # ========== EMBEDDINGS ==========
    # Calcolo embedding per ogni chunk
    for i, chunk in enumerate(chunks):
        emb = embedding.get_embedding(chunk)
        emb_json = json.dumps(emb)
        chunk_records.append((numero, i, cliente, titolo, autore, documento, url_doc, chunk, emb_json))
        corpus.append(chunk)

    cursor.execute("UPDATE VAR_RICSW SET BooleanValue = 1 WHERE VariableName = 'ELABORATO' AND InstanceID = ?", (instance_id))
    # Salvataggio di tutti i chunk di un documento in un'unica query per maggior efficienza
    if chunk_records:
        cursor.executemany(
            "INSERT INTO DocumentChunks (NumRI, Progressivo, Cliente, Titolo, Autore, Documento, Url_doc, Content, Embedding)" \
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, CAST(CAST(? AS VARCHAR(MAX)) AS VECTOR(1536)))", 
            chunk_records
        )
    conn.commit()
    print(f"Embedding creati per file: {numero}{extension} ,\t{len(chunks)} chunk generati\n{'-'*40}.\n")


# ========== INDEXING BM25 ==========
print(f"\n{'-'*40}\nRicostruzione completa indice BM25...")

language = 'italian'
stemmer = SnowballStemmer(language)
stop_words = stopwords.words(language)

# ---- Carica TUTTI i chunk dal DB ----
cursor = conn.cursor()
cursor.execute("SELECT Content FROM DocumentChunks")
all_chunks = [row[0] for row in cursor.fetchall()]

print(f"Totale chunk nel DB: {len(all_chunks)}")

if not all_chunks:
    print("Nessun chunk presente nel DB. Indice non creato.")
else:
    all_tokens = bm25s.tokenize(
        all_chunks,
        stopwords=stop_words,
        stemmer=lambda tokens: [stemmer.stem(t.lower()) for t in tokens]
    )

    # ---- Creazione e salvataggio reverse index ----
    retriever = bm25s.BM25()
    retriever.index(all_tokens)

    index_folder = "reverse_index"
    os.makedirs(index_folder, exist_ok=True)
    index_path = os.path.join(index_folder, "bm25_index")
    retriever.save(index_path)

    print(f"Documenti indicizzati: {len(retriever.corpus)}")
    print(f"Token unici nel vocabolario: {len(retriever.vocab)}")

