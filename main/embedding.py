from openai import AzureOpenAI
import os

# Calcola gli embeddings per ogni documento a partire dal testo estratto
def get_embedding(text):

#
#

    # Inizializza la chiamata API di Azure OpenAI
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_API_KEY"),
        azure_endpoint=os.getenv("/api/v{version}/OpenAi/Embeddings"),
        api_version=os.getenv("2023-05-15")
    )

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding