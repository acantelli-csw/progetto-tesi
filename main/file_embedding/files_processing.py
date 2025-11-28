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
SELECT top 3 v.InstanceID,
    MAX(CASE WHEN v.VariableName = 'NUMERO' THEN v.StringValue END) AS numero,
    MAX(CASE WHEN v.VariableName = 'CLIENTE' THEN v.StringValue END) AS cliente,
    MAX(CASE WHEN v.VariableName = 'TITOLO' THEN v.StringValue END) AS titolo,
    MAX(CASE WHEN v.VariableName = 'AUTORE' THEN v.StringValue END) AS autore,
    MAX(CASE WHEN v.VariableName = 'DOCUMENTO' THEN v.StringValue END) AS documento,
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
ORDER BY v.InstanceID;
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
            "INSERT INTO DocumentChunks (NumRI, Progressivo, Cliente, Titolo, Autore, Doc, Url_doc, Content, Embedding)" \
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, CAST(CAST(? AS VARCHAR(MAX)) AS VECTOR(1536)))", 
            chunk_records
        )
    conn.commit()
    print(f"Embedding creati per file: {numero}{extension} ,\t{len(chunks)} chunk generati\n{'-'*40}.\n")

    
# ========== INDEXING BM25 ==========
if corpus:
    print(f"\n{'-'*40}\nCreazione indice BM25 per {len(corpus)} chunk...")

    language = 'italian'
    stemmer = SnowballStemmer(language)
    stop_words = stopwords.words(language)
    
    # Tokenizzazione con stemmer
    corpus_tokens = bm25s.tokenize(
        corpus, 
        stopwords=stop_words, 
        stemmer=stemmer.stem
    )
    
    # Creazione e indicizzazione BM25
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    # Salvataggio dell'indice in locale
    index_folder = "reverse_index"
    os.makedirs(index_folder, exist_ok=True)
    index_path = os.path.join(index_folder, "bm25_index")
    retriever.save(index_path)
    
    print(f"Indice BM25 creato e salvato in '{index_path}'")
    print(f"Documenti indicizzati: {len(corpus)}")
    print(f"Token unici nel vocabolario: {len(corpus_tokens.vocab)}")
else:
    print("Nessun chunk da indicizzare.")

cursor.close()
conn.close()
print(f"\n{'-'*40}\nTutti i file sono stati processati!")
