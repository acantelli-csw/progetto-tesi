from llama_index import GPTVectorStoreIndex, ServiceContext, LLMPredictor
from langchain_openai import AzureChatOpenAI
from llama_index.readers.schema.base import Document as LlamaDocument
import sys
import os
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from file_embedding.db_connection import get_connection
from llama_index.core.schema import Document as LlamaDocument

def load_all_docs():
    cursor = get_connection().cursor()
    cursor.execute("""
        SELECT ID, NumRI, Progressivo, Cliente, Titolo, Autore, Documento, Url_doc, Content
        FROM DocumentChunks 
    """)
    rows = cursor.fetchall()
    
    documents = []
    for row in rows:
        doc_id, numero, progressivo, cliente, titolo, autore, documento, url_doc, content = row

        # Crea un Document di LangChain
        doc = Document(
            page_content=content,
            metadata={
                "id": doc_id,
                "numero": numero,
                "progressivo": progressivo,
                "cliente": cliente,
                "titolo": titolo,
                "autore": autore,
                "documento": documento,
                "url_doc": url_doc,
            }
        )
        documents.append(doc)
    return documents

# Carica le variabili d'ambiente
load_dotenv()
# 1️⃣ Configura Azure LLM
azure_llm = AzureChatOpenAI(
    azure_endpoint="https://cs-test.openai.azure.com",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("LLM_VERSION"),
    deployment_name=os.getenv("LLM_MODEL"),
    temperature=0.3,
)

llm_predictor = LLMPredictor(llm=azure_llm)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# 2️⃣ Carica documenti dal DB o cartella
docs = []
for i, d in enumerate(load_all_docs()):  # la tua funzione
    if d.page_content.strip():
        docs.append(LlamaDocument(text=d.page_content, metadata=d.metadata, id_=f"doc_{i}"))

# 3️⃣ Crea indice vettoriale
index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)

# 4️⃣ Esempio: query sull’indice
query = "Genera 5 domande per testare un sistema RAG su questi documenti."
response = index.as_query_engine().query(query)
print(response)




