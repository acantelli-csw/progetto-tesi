
# PSEUDO-CODICE
# 1. Leggi tutti i documenti dal DB


# 2. Calcola gli embeddings per ogni documento

from openai import OpenAI
client = OpenAI(api_key="YOUR_API_KEY")

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=document_text
)
embedding = response.data[0].embedding


# 3. Salva gli embeddings nel DB

cursor.execute("""
    UPDATE Documents SET embedding = ? WHERE id = ?
""", (embedding, doc_id))
conn.commit()