from PIL import Image
from io import BytesIO
from docx import Document
import textract
import easyocr
import tempfile
import os
import fitz  

# Estrai testo e immagini dai documenti in base all'estensione
def extract_text_from_varbinary(file_bytes, extension):

    ext = extension.lower()
    full_text = ""
    # Inizializza il reader OCR
    reader = easyocr.Reader(['it', 'en'])

    try:

        # Caso 1: PDF (testo + OCR)
        if ext == ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            # --- Testo nativo ---
            with fitz.open(tmp_path) as pdf:
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
                        
                        result = reader.readtext(img)
                        ocr_text = " ".join([res[1] for res in result])

                        if ocr_text.strip():
                            full_text += f"\n[OCR p.{page_index+1} img.{img_index+1}]: {ocr_text}\n"

            os.remove(tmp_path)


        # Caso 2: DOCX (testo + OCR immagini)
        elif ext == ".docx":
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
    
                    result = reader.readtext(img)
                    ocr_text = " ".join([res[1] for res in result])

                    if ocr_text.strip():
                        full_text += f"\n[OCR immagine]: {ocr_text}\n"

            os.remove(tmp_path)

        # Caso 3: DOC (conversione + testo)
        elif ext == ".doc":
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

        # Caso non gestito
        else:
            print(f"Estensione non gestita: {ext}")
            full_text = ""

        # Pulizia testo
        full_text = full_text.strip().replace("\x0c", "")
        return full_text

    except Exception as e:
        print(f"Errore elaborando file .{ext}: {e}")
        return ""
