import sys
import os
import json
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from file_embedding.db_connection import get_connection
from ragas.testset import TestsetGenerator
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings  # ✅ Usa Azure
from llama_index.core.schema import Document as LlamaDocument

def load_all_docs():
    """Carica tutti i documenti dal database e li converte in formato Ragas"""
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

def generate_testset(n_samples=5, output_file="testset_ragas.json", use_custom_retriever=False):
    """
    Genera il testset usando Ragas con Azure OpenAI
    
    Args:
        n_samples: Numero di domande da generare
        output_file: File di output per salvare il testset
        use_custom_retriever: Se True, arricchisce il testset con i documenti recuperati
    """
    print("📚 Caricamento documenti...")
    documents = load_all_docs()
    print(f"✅ Caricati {len(documents)} documenti")
    
    # Converti in LlamaIndex Documents
    print("🔄 Conversione documenti in formato LlamaIndex...")
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
    
    print(f"📄 Documenti validi: {len(llama_docs)}")
    
    if len(llama_docs) < 5:
        print(f"❌ Troppo pochi documenti ({len(llama_docs)}). Servono almeno 5 documenti validi.")
        return None

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

    # Inizializza il generator di Ragas
    generator = TestsetGenerator(
        llm=LangchainLLMWrapper(azure_llm), 
        embedding_model=LangchainEmbeddingsWrapper(embeddings=azure_embeddings),
    )
    
    print(f"🎯 Generazione di {n_samples} domande da {len(llama_docs)} documenti...")
    try:
        # ✅ Usa almeno 10-20 documenti per generare personas diverse
        num_docs = min(30, len(llama_docs))  # Usa max 30 documenti
        print(f"   Usando {num_docs} documenti per la generazione...")
        
        # Genera il testset
        testset = generator.generate_with_llamaindex_docs(
            documents=llama_docs[:num_docs],
            testset_size=n_samples,
            raise_exceptions=True,
        )
        
        print(f"✅ Testset generato! Numero domande: {len(testset)}")
        
        # Converti in formato serializzabile
        testset_dict = testset.to_list()
        
        # Salva il testset
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(testset_dict, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Testset salvato in: {output_file}")
        
        # Mostra statistiche
        print("\n📊 Statistiche testset:")
        if 'examples' in testset_dict:
            examples = testset_dict['examples']
            print(f"  - Totale esempi: {len(examples)}")
            
            # Conta tipi di domande
            question_types = {}
            for ex in examples:
                q_type = ex.get('type', 'unknown')
                question_types[q_type] = question_types.get(q_type, 0) + 1
            
            print("  - Distribuzione tipi:")
            for q_type, count in question_types.items():
                print(f"    • {q_type}: {count}")
        
        return testset_dict
        
    except Exception as e:
        print(f"❌ Errore: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Genera il testset
    testset = generate_testset(n_samples=5)
    
    if testset:
        print("\n✨ Testset generato con successo!")
        print("Puoi ora usarlo per valutare il tuo sistema RAG.")
    else:
        print("\n⚠️ Generazione testset fallita. Controlla gli errori sopra.")