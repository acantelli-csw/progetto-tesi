from dotenv import load_dotenv, find_dotenv
import pyodbc
import os

# Cerca il .env risalendo l'albero delle directory a partire da questo file.
# Questo garantisce che le variabili vengano caricate correttamente
# indipendentemente da quale cartella viene usata come working directory
# quando si lancia lo script (es. main/, main/evaluation/, ecc.).
load_dotenv(find_dotenv())

def get_connection():
    conn_str = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=CSW-021;"
        "DATABASE=BPM;"
        f"UID={os.getenv('DB_UID')};"
        f"PWD={os.getenv('DB_PASSWORD')};"
        "Encrypt=optional;"
        "TrustServerCertificate=yes;"
    )

    try:
        conn = pyodbc.connect(conn_str)
        print("Connessione al DB avvenuta con successo!")
        return conn
    except Exception as e:
        print("Errore di connessione:", e)
        raise  # Rilancia l'eccezione invece di ritornare conn non assegnata