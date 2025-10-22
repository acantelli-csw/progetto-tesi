import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from extract_text import extract_text_from_varbinary
import db_connection
import embedding

# Elaborazione file - TESTO + IMMAGINI

def process_files(chunk_size = 1000, chunk_overlap = 150):

    # Ottieni la connessione al DB + estrazione file senza embedding
    conn = db_connection.get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, filename, extension, filedata FROM Files WHERE embedding IS NULL")
    rows = cursor.fetchall()

    # Elaborazione di ogni file
    for row in rows:
        file_id, filename, ext, blob = row

        # Estrazione testo + OCR immagini interne
        text = extract_text_from_varbinary(blob, ext)
        if not text.strip():
            print(f"Nessun testo estratto dal file: {filename}")
            continue

        # Suddivisione in chunk
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(text)

        # Calcolo embedding per ogni chunk
        embeddings = [embedding.get_embedding(chunk) for chunk in chunks]
        

        # TODO: non aggregare i chunk, ma salvarli in una tabella separata per retrieval + ranking
        # Media dei chunk per avere un unico embedding per documento
        avg_embedding = [sum(col)/len(col) for col in zip(*embeddings)]


        # TODO: Invece di usare json.dumps, utilizzare il formato nativo di SQL Server per vettori (versione 2025+)
        # Salvataggio embedding nel DB
        cursor.execute(
            "UPDATE Files SET embedding = ? WHERE id = ?",
            (json.dumps(avg_embedding), file_id)
        )
        conn.commit()

        print(f"File processato: {filename}, {len(chunks)} chunk generati")

    cursor.close()
    conn.close()
    print("Tutti i file sono stati processati!")
