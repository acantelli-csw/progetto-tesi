import sys
import os
import json
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from file_embedding.db_connection import get_connection
from ragas.testset import TestsetGenerator
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from llama_index.core.schema import Document as LlamaDocument

# Carica le variabili d'ambiente
load_dotenv()

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

def generate_testset(n_samples=30, output_file="testset_ragas.json"):

    documents = load_all_docs()
    print(f"✅ Documenti caricati: {len(documents)}")
    
    # Converti in LlamaIndex Documents
    llama_docs = []
    for i, doc in enumerate(documents):
        # Filtra documenti troppo corti o vuoti
        if doc.page_content and len(doc.page_content.strip()) > 50:
            llama_doc = LlamaDocument(
                text=doc.page_content,
                metadata=doc.metadata if hasattr(doc, 'metadata') else {},
                id_=f"doc_{i}"
            )
            llama_docs.append(llama_doc)
    print(f"📄 Documenti convertiti in formato LlamaIndex validi: {len(llama_docs)}")

    azure_llm = AzureChatOpenAI(
        azure_endpoint="https://cs-test.openai.azure.com",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("LLM_VERSION"),
        deployment_name=os.getenv("LLM_MODEL"),
        temperature=0.3,
    )

    azure_embeddings = AzureOpenAIEmbeddings(
        azure_endpoint="https://cs-test.openai.azure.com",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("EMBEDDING_VERSION_2"),
        deployment=os.getenv("EMBEDDING_MODEL_2"),
    )

    # Inizializzazione del generator di Ragas
    generator = TestsetGenerator(
        llm=LangchainLLMWrapper(azure_llm), 
        embedding_model=LangchainEmbeddingsWrapper(embeddings=azure_embeddings),
    )
    
    try:
        num_docs = min(50, len(llama_docs))
        print(f"🎯 Generazione di {n_samples} domande da {num_docs} documenti...")

        testset = generator.generate_with_llamaindex_docs(
            documents=llama_docs[:num_docs],
            testset_size=n_samples,
            raise_exceptions=True,
        )
        print(f"✅ Testset generato! Numero domande: {len(testset)}")
        
        testset_list = testset.to_list()
        
        # Salva il testset
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(testset_list, f, ensure_ascii=False, indent=2)
        print(f"💾 Testset salvato in: {output_file}")
        
        return testset_list
        
    except Exception as e:
        print(f"❌ Errore: {str(e)}")
        return None


if __name__ == "__main__":    
    testset = generate_testset(n_samples=30)
    if testset:
        print("\n✨ Testset generato con successo!")
        print("Puoi ora usarlo per valutare il tuo sistema RAG.")
    else:
        print("\n⚠️ Generazione testset fallita. Controlla gli errori sopra.")