from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import extract_text
import db_connection
import embedding

chunk_size = 1000, chunk_overlap = 150

# Ottieni la connessione al DB + estrazione file senza embedding
conn = db_connection.get_connection()
cursor = conn.cursor()

# Query per ottenere i file da processare e le loro variabili associate
query = """
SELECT v.InstanceID,
       MAX(CASE WHEN v.VariableName = 'cliente' THEN v.StringValue END) AS cliente,
       MAX(CASE WHEN v.VariableName = 'numero' THEN v.StringValue END) AS numero,
       MAX(CASE WHEN v.VariableName = 'titolo' THEN v.StringValue END) AS titolo,
       MAX(CASE WHEN v.VariableName = 'autore' THEN v.StringValue END) AS autore,
       MAX(f.FileData) AS FileData
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

# Elaborazione di ogni file
for row in rows:

    # TODO fix importing and add extension detection
    instance_id, cliente, numero, titolo, autore, file_data = row

    # Estrazione testo + OCR immagini interne
    text = extract_text.extract_text_from_varbinary(file_data, '.docx')
    if not text.strip():
        print(f"Nessun testo estratto dal file con InstanceID {instance_id}, salto il file.")
        continue


    # Suddivisione in chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)

    # Calcolo e salvataggio su DB degli embedding per ogni chunk
    chunk_records = []
    for i, chunk in enumerate(chunks):
        emb = embedding.get_embedding(chunk)
        chunk_records.append((file_id, i, chunk, emb))

    # Salvataggio di tutti i chunk in un'unica query
    cursor.executemany(
        "INSERT INTO FileChunks (file_id, chunk_index, chunk_text, chunk_embedding) VALUES (?, ?, ?, ?)",
        chunk_records
    )

    conn.commit()
    print(f"File processato: {filename}, {len(chunks)} chunk generati")

cursor.close()
conn.close()
print("Tutti i file sono stati processati!")
