import json
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
from io import BytesIO
import db_connection
import pytesseract
import textract


# Leggi tutti i documenti dal DB
conn = db_connection.get_connection()
cursor = conn.cursor()

# TODO: fix the query to fetch documents
# cursor.execute("SELECT id, filename, extension, filedata FROM Files WHERE embedding IS NULL")
cursor.execute("SELECT ID, StringValue FROM VAR_INVIO_EMAIL_VALUTAZIONE WHERE VariableName = 'Email'")
rows = cursor.fetchall()

# for r in rows:
#     print(r)


# Estrai testo dai documenti in base all'estensione
# Tipi di file all'interno del DB Server Demo ad oggi: .doc, .docx, .ini, .jpg, .pdf, .png, .txt
def extract_text_from_varbinary(file_blob, extension):
    ext = extension.lower()

    if ext == ".pdf":
        with open("temp.pdf", "wb") as f:
            f.write(file_blob)
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        return " ".join([d.page_content for d in docs])

    elif ext == ".docx":
        with open("temp.docx", "wb") as f:
            f.write(file_blob)
        loader = Docx2txtLoader("temp.docx")
        docs = loader.load()
        return " ".join([d.page_content for d in docs])

    elif ext == ".doc":
        with open("temp.doc", "wb") as f:
            f.write(file_blob)
        text = textract.process("temp.doc").decode("utf-8")
        return text

    elif ext in [".txt", ".ini"]:
        return file_blob.decode("utf-8")

    elif ext in [".jpg", ".jpeg", ".png"]:
        img = Image.open(BytesIO(file_blob))
        return pytesseract.image_to_string(img)

    else:
        return ""
    

# Chunking e generazione embedding
import embedding
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

for row in rows:
    file_id, filename, ext, blob = row
    text = extract_text_from_varbinary(blob, ext)
    if not text.strip():
        print(f"Nessun testo estratto dal file {filename}")
        continue

    chunks = splitter.split_text(text)
    embeddings = [embedding.get_embedding(chunk) for chunk in chunks]
    avg_embedding = [sum(col)/len(col) for col in zip(*embeddings)]

    # TODO: Invece di usare l'embedding medio, salvare gli embeddings dei singoli chunk in una tabella separata
    # TODO: Invece di usare json.dumps, utilizzare il formato nativo di SQL Server per vettori (versione 2025+)

# Salvataggio embedding nel DB
    cursor.execute("UPDATE Files SET embedding = ? WHERE id = ?", (json.dumps(avg_embedding), file_id))
    conn.commit()
    print(f"Inserito embedding per file ID {file_id}")
