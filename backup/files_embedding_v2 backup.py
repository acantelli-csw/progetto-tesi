import json
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
from io import BytesIO
import db_connection
import pytesseract
import textract
import pytesseract
import tempfile
import os
import fitz
import pdfplumber
from docx import Document

# TODO: Gestione seprata delle immagini all'interno dei documenti

# Leggi tutti i documenti dal DB
conn = db_connection.get_connection()
cursor = conn.cursor()

cursor.execute("SELECT id, filename, extension, filedata FROM Files WHERE embedding IS NULL")
rows = cursor.fetchall()

# Elaborazione file + salvataggio embedding
for row in rows:
    file_id, filename, ext, file_blob = row
    print(f"Elaborando file ID {file_id}: {filename}")

    embedding = generate_embeddings_from_file(file_blob, ext)
    if not embedding:
        print(f"Nessun embedding generato per {filename}")
        continue

    cursor.execute(
        "UPDATE Files SET embedding = ? WHERE id = ?",
        (json.dumps(embedding), file_id)
    )

    conn.commit()
    print(f"Embedding salvato per {filename}")

conn.close()
print("Processo completato")


def extract_text_from_varbinary(file_bytes, extension):

    ext = extension.lower()
    full_text = ""

    try:
        # ---------------------------------------------------------
        # Caso 1: PDF (testo + OCR)
        # ---------------------------------------------------------
        if ext == "pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            # --- Testo nativo ---
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"

            # --- OCR su immagini ---
            with fitz.open(tmp_path) as doc:
                for page_index, page in enumerate(doc):
                    for img_index, img in enumerate(page.get_images(full=True)):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        img = Image.open(BytesIO(image_bytes))
                        ocr_text = pytesseract.image_to_string(img)
                        if ocr_text.strip():
                            full_text += f"\n[OCR p.{page_index+1} img.{img_index+1}]: {ocr_text}\n"

            os.remove(tmp_path)

        # ---------------------------------------------------------
        # Caso 2: DOCX (testo + OCR immagini)
        # ---------------------------------------------------------
        elif ext == "docx":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            doc = Document(tmp_path)
            # --- Testo nativo ---
            for p in doc.paragraphs:
                full_text += p.text + "\n"

            # --- OCR su immagini embedded ---
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    img_data = rel.target_part.blob
                    img = Image.open(BytesIO(img_data))
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        full_text += f"\n[OCR immagine]: {ocr_text}\n"

            os.remove(tmp_path)

        # ---------------------------------------------------------
        # Caso 3: DOC (conversione + testo)
        # ---------------------------------------------------------
        elif ext == "doc":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                text = textract.process(tmp_path).decode("utf-8")
                full_text += text
            except Exception as e:
                print(f"Errore estrazione DOC: {e}")
            finally:
                os.remove(tmp_path)

        # ---------------------------------------------------------
        # Caso 4: TXT o INI (lettura diretta)
        # ---------------------------------------------------------
        elif ext in ["txt", "ini"]:
            try:
                full_text = file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                # fallback per altri encoding
                full_text = file_bytes.decode("latin-1", errors="ignore")

        # ---------------------------------------------------------
        # Caso 5: Immagini (JPG, JPEG, PNG)
        # ---------------------------------------------------------
        elif ext in ["jpg", "jpeg", "png"]:
            img = Image.open(BytesIO(file_bytes))
            ocr_text = pytesseract.image_to_string(img)
            full_text += ocr_text

        # ---------------------------------------------------------
        # Caso non gestito
        # ---------------------------------------------------------
        else:
            print(f"⚠️ Estensione non gestita: {ext}")
            full_text = ""

        # Pulizia testo
        full_text = full_text.strip().replace("\x0c", "")
        return full_text

    except Exception as e:
        print(f"❌ Errore elaborando file .{ext}: {e}")
        return ""


cursor.execute("SELECT id, filename, extension, filedata FROM Files WHERE embedding IS NULL")
rows = cursor.fetchall()

for row in rows:
    file_id, filename, ext, blob = row
    text = extract_text_from_varbinary(blob, ext)

    if not text.strip():
        print(f"Nessun testo estratto da {filename}")
        continue

    # → Chunking + Embedding + Update DB
