import extract_text
import easyocr
import os

# Inizializza il reader OCR una volta sola
reader = easyocr.Reader(['it', 'en'], gpu=False)

def processa_documenti(cartella):
    # Ottieni tutti i file .doc nella cartella
    files = [f for f in os.listdir(cartella) if f.lower().endswith(".doc")]

    contatore = 1

    for nome_file in files:
        percorso = os.path.join(cartella, nome_file)
        
        # --- LEGGI IL FILE COME BYTES ---
        with open(percorso, "rb") as f:
            file_bytes = f.read()

        # Estrai estensione
        _, ext = os.path.splitext(nome_file)

        # Chiama la tua funzione
        testo = extract_text.extract_text_from_varbinary(
            file_data=file_bytes,
            extension=ext,
            numero=contatore,
            reader=reader
        )

        print(f"File elaborato n° {contatore}: {nome_file}")
        #print(f"Testo: {testo}\n")
        contatore += 1

# ▶ Esempio d’uso
processa_documenti("C:/BPM/RI/DOC/")

