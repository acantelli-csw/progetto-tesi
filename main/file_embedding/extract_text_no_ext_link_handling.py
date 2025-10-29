from PIL import Image
from io import BytesIO
from docx import Document
import numpy as np
import textract
import tempfile
import os
import fitz  


# Estrai testo e immagini dai documenti in base all'estensione
def extract_text_from_varbinary(file_data, extension, numero, reader):

    ext = extension.lower()
    full_text = ""

    try:

        # Caso 1: PDF (testo + OCR)
        if ext == ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_data)
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
                        img_np = np.array(img)
                        result = reader.readtext(img_np)

                        ocr_text = " ".join([res[1] for res in result])

                        if ocr_text.strip():
                            full_text += f"\n[OCR p.{page_index+1} img.{img_index+1}]: {ocr_text}\n"

            os.remove(tmp_path)


        # Caso 2: DOCX (testo + OCR immagini)
        elif ext == ".docx":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name

            doc = Document(tmp_path)

            # --- Testo nativo ---
            for p in doc.paragraphs:
                full_text += p.text + "\n"

            # --- OCR su immagini embedded ---
            for rel in doc.part.rels.values():
                target_ref = getattr(rel, "target_ref", "")
                target_part = getattr(rel, "target_part", None)

                # Se la relazione punta a un'immagine
                if "image" in target_ref:
                    # Check se è un link esterno
                    is_external = hasattr(rel, "is_external") and rel.is_external
                    
                    if is_external or target_part is None:
                        print(f"Ignorata immagine con link esterno: {target_ref}")
                        #continue  # salta immagine non incorporata

                    try:
                        img_data = target_part.blob
                        img = Image.open(BytesIO(img_data))
                        img_np = np.array(img)
                        result = reader.readtext(img_np)
                        ocr_text = " ".join([res[1] for res in result])
                        if ocr_text.strip():
                            full_text += f"\n[OCR immagine embedded]: {ocr_text}\n"
                    except Exception as e:
                        print(f"Errore OCR immagine DOCX: {e}")
                        
            os.remove(tmp_path)

        # Caso 3: DOC (conversione + testo)
        elif ext == ".doc":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as tmp:
                tmp.write(file_data)
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
        print(f"Errore elaborando file {numero}{ext}: {e}")
        return ""