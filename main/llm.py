from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env")

# Calcola gli embeddings per ogni documento a partire dal testo estratto
def llm_answer(text):

    # Configura client per chiamata API di Azure OpenAI
    client = AzureOpenAI(
        azure_endpoint = os.getenv("LLM_URL"),
        api_key = os.getenv("OPENAI_API_KEY"),
        api_version = os.getenv("LLM_MODEL")
    )

    # Esegui chiamata embedding per chat completion
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[
            {"role": "system", "content": "Sei un assistente utile e conciso."},
            {"role": "user", "content": text}
        ],
        temperature=0.7,
        max_tokens=500
    )

    # Estrai la risposta del modello
    answer = response.choices[0].message.content
    print("Risposta GPT-4.1:\n", answer)