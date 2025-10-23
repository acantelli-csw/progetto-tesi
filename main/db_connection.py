from dotenv import load_dotenv
import pyodbc
import os

load_dotenv(dotenv_path=".env")

def get_connection():
    conn_str = (
        "DRIVER={SQL Server};"
        "SERVER=CSW-021;"                               # nome o IP del server
        "DATABASE=BPM;"                                 # nome del DB
        f"UID={os.getenv("DB_UID")};"                   # SQL login
        f"PWD={os.getenv("DB_PASSWORD")};"              # password SQL
        "Encrypt=optional;"                     
        "TrustServerCertificate=yes;"
    #   "Connection Timeout=30;"
    )

    try:
        conn = pyodbc.connect(conn_str)
        print("Connessione avvenuta con successo!")
    except Exception as e:
        print("Errore di connessione:", e)

    return conn