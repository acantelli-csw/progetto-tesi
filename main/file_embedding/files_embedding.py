import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
from langchain_text_splitters import RecursiveCharacterTextSplitter
import extract_text
import db_connection
import embedding
import json

# TODO: tune these parameters
chunk_size = 1000
chunk_overlap = 150

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
ORDER BY v.InstanceID;
"""

cursor.execute(query)

rows = cursor.fetchall()

# Elaborazione di ogni file
for row in rows:

    instance_id, cliente, numero, titolo, autore, file_data, extension = row

    # Estrazione testo + OCR immagini interne
    text = extract_text.extract_text_from_varbinary(file_data, extension, numero)
    if not text.strip():
        print(f"Nessun testo estratto dal file {numero}{extension}, salto il file.")
        continue

    # Suddivisione in chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)

    # Calcolo e salvataggio su DB degli embedding per ogni chunk
    chunk_records = []
    for i, chunk in enumerate(chunks):
        emb = embedding.get_embedding(chunk)
        emb_json = json.dumps(emb)
        chunk_records.append((numero, i, titolo, cliente, autore, chunk, emb_json))


    # Salvataggio di tutti i chunk in un'unica query per maggior efficienza
    cursor.executemany(
        "INSERT INTO DocumentChunks (NumRI, Progressivo, TitoloRI, Cliente, Autore, Content, Embedding)" \
        "VALUES (?, ?, ?, ?, ?, ?, CAST(CAST(? AS VARCHAR(MAX)) AS VECTOR(1536)))",
        chunk_records
    )

    cursor.execute("UPDATE VAR_RICSW SET BooleanValue = 1 WHERE VariableName = 'elaborato' AND InstanceID = ?", (instance_id))

    conn.commit()
    print(f"File processato: {numero}{extension},\t{len(chunks)} chunk generati\n{'-'*40}\n")

cursor.close()
conn.close()
print("Tutti i file sono stati processati!")
