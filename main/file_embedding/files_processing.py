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
SELECT v.InstanceID,
    MAX(CASE WHEN v.VariableName = 'cliente' THEN v.StringValue END) AS Cliente,
    MAX(CASE WHEN v.VariableName = 'numero' THEN v.StringValue END) AS Numero,
    MAX(CASE WHEN v.VariableName = 'titolo' THEN v.StringValue END) AS Titolo,
    MAX(CASE WHEN v.VariableName = 'autore' THEN v.StringValue END) AS Autore,
    MAX(f.FileData) AS FileData,
    MAX(f.Extension) AS Extension
FROM VAR_RICSW v 
JOIN DocumentFiles f
    ON v.InstanceID = f.InstanceID
WHERE v.InstanceID IN (
    SELECT v2.InstanceID
    FROM VAR_RICSW v2
    WHERE v2.VariableName = 'elaborato'
        AND v2.BooleanValue = 0
  )
GROUP BY v.InstanceID
ORDER BY Numero;
"""

cursor.execute(query)
rows = cursor.fetchall()

corpus = []
chunk_records = []

# Elaborazione file estratti
for row in rows:

    instance_id, cliente, numero, titolo, autore, file_data, extension = row

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
        chunk_records.append((numero, i, titolo, cliente, autore, chunk, emb_json))
        corpus.append(chunk)

    cursor.execute("UPDATE VAR_RICSW SET BooleanValue = 1 WHERE VariableName = 'elaborato' AND InstanceID = ?", (instance_id))
    print(f"Embedding creati per file: {numero}{extension} ,\t{len(chunks)} chunk generati\n{'-'*40}.\n")

# Salvataggio di tutti i chunk in un'unica query per maggior efficienza
if chunk_records:
    cursor.executemany(
        "INSERT INTO DocumentChunks (NumRI, Progressivo, TitoloRI, Cliente, Autore, Content, Embedding)" \
        "VALUES (?, ?, ?, ?, ?, ?, CAST(CAST(? AS VARCHAR(MAX)) AS VECTOR(1536)))", 
        chunk_records
    )
    conn.commit()
    
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
