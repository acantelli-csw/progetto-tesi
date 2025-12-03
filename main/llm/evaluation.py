import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from file_embedding.db_connection import get_connection
import search
import llm

# Import corretti per Ragas
from ragas.testset import TestsetGenerator
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.graph import KnowledgeGraph, Node
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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
        
        # Crea un Document di LangChain (formato richiesto da Ragas)
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

def create_custom_retriever(retrieve_function):
    """
    Wrapper per adattare la tua funzione di retrieval al formato Ragas.
    Ragas si aspetta un callable che accetta una query string e ritorna lista di Document.
    """
    def wrapped_retriever(query: str) -> list[Document]:
        # Usa la tua logica di retrieval
        selected_docs = retrieve_function(query)
        
        # Converti i tuoi documenti in formato Document se necessario
        langchain_docs = []
        for doc in selected_docs:
            if isinstance(doc, Document):
                langchain_docs.append(doc)
            elif isinstance(doc, dict):
                # Se sono dizionari, convertili
                content = doc.get("content", "")
                metadata = {k: v for k, v in doc.items() if k != "content"}
                langchain_docs.append(Document(page_content=content, metadata=metadata))
            else:
                print(f"Formato documento non riconosciuto: {type(doc)}")
        
        return langchain_docs
    
    return wrapped_retriever

def retrieve_fn(user_prompt: str):
    """Funzione di retrieval personalizzata"""
    # 1 - Decisione strumenti
    tools = llm.decide_tools(user_prompt)

    # 2 - Recupero documenti
    all_documents = []
    if tools["use_semantic"]:
        print("\nUso la ricerca SEMANTICA")
        all_documents += search.semantic_search(user_prompt, 25)

    if tools["use_keyword"]:
        print("\nUso la ricerca per KEYWORDS")
        all_documents += search.keyword_search(user_prompt, 25)

    # 3 - Selezione documenti in base alla coerenza
    selected_docs = []
    if tools["use_semantic"] or tools["use_keyword"]:
        if all_documents:
            document_selection = llm.select_documents(user_prompt, all_documents)
            # Selezione documenti rilevanti
            selected_docs = [all_documents[i] for i in document_selection['relevant_docs']]
    
    return selected_docs

def enrich_testset_with_retriever(testset, retriever_fn):
    """
    Arricchisce il testset usando il tuo retriever custom.
    Per ogni domanda generata, recupera i documenti che il TUO sistema recupererebbe.
    Questo ti permette di valutare quanto bene il tuo retriever funziona.
    """
    enriched_examples = []
    
    testset_dict = testset.to_dict()
    examples = testset_dict.get('examples', [])
    
    for i, example in enumerate(examples):
        question = example.get('question', '')
        
        if question:
            print(f"  Processing {i+1}/{len(examples)}: {question[:60]}...")
            
            # Usa il TUO retriever per recuperare documenti
            try:
                retrieved_docs = retriever_fn(question)
                
                # Aggiungi i documenti recuperati all'esempio
                example['retrieved_contexts'] = [
                    doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    for doc in retrieved_docs
                ]
                example['retrieved_count'] = len(retrieved_docs)
                
            except Exception as e:
                print(f"    ⚠️ Errore nel retrieval: {e}")
                example['retrieved_contexts'] = []
                example['retrieved_count'] = 0
        
        enriched_examples.append(example)
    
    # Aggiorna il testset
    testset_dict['examples'] = enriched_examples
    
    return testset_dict

def generate_testset(n_samples=50, output_file="testset_ragas.json", use_custom_retriever=True):
    """
    Genera il testset usando Ragas con componenti custom
    
    Args:
        n_samples: Numero di domande da generare
        output_file: File di output per salvare il testset
        use_custom_retriever: Se True, arricchisce il testset con i documenti recuperati dal tuo retriever
    """
    print("📚 Caricamento documenti...")
    documents = load_all_docs()
    print(f"✅ Caricati {len(documents)} documenti")

    # Crea il retriever wrapper
    custom_retriever = create_custom_retriever(retrieve_fn)

    # Inizializza il generator con la nuova API
    generator = TestsetGenerator(
        llm=ChatOpenAI(model="gpt-4", temperature=0.3),
        knowledge_graph=KnowledgeGraph(),  # Opzionale: per relazioni tra documenti
        embedding_model=LangchainEmbeddingsWrapper(embeddings=OpenAIEmbeddings(model="text-embedding-3-small")),
    )
    
    print(f"🎯 Generazione di {n_samples} domande...")
    try:
        # Genera il testset
        testset = generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=n_samples,
            raise_exceptions=False  # Continua anche se alcune domande falliscono
        )
        
        # Arricchisco il testset con i documenti che il TUO retriever recupererebbe
        if use_custom_retriever:
            print("\n🔍 Arricchimento testset con custom retriever...")
            testset = enrich_testset_with_retriever(testset, custom_retriever)
            print("✅ Testset arricchito con retrieval custom")
        
        print(f"✅ Testset generato! Numero domande: {len(testset)}")
        
        # Converti in formato serializzabile (già fatto se use_custom_retriever=True)
        if not use_custom_retriever:
            testset = testset.to_dict()
        
        # Salva il testset
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(testset, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Testset salvato in: {output_file}")
        
        # Mostra alcune statistiche
        print("\n📊 Statistiche testset:")
        if 'examples' in testset:
            examples = testset['examples']
            print(f"  - Totale esempi: {len(examples)}")
            
            # Conta tipi di domande
            question_types = {}
            for ex in examples:
                q_type = ex.get('type', 'unknown')
                question_types[q_type] = question_types.get(q_type, 0) + 1
            
            print("  - Distribuzione tipi:")
            for q_type, count in question_types.items():
                print(f"    • {q_type}: {count}")
            
            # Se hai usato il custom retriever, mostra anche stats sul retrieval
            if use_custom_retriever and examples:
                retrieved_counts = [ex.get('retrieved_count', 0) for ex in examples]
                avg_retrieved = sum(retrieved_counts) / len(retrieved_counts) if retrieved_counts else 0
                print(f"\n  - Media documenti recuperati: {avg_retrieved:.2f}")
                print(f"  - Min/Max documenti: {min(retrieved_counts)}/{max(retrieved_counts)}")
        
        return testset
        
    except Exception as e:
        print(f"❌ Errore durante la generazione: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_with_testset(testset_file="testset_ragas.json"):
    """
    Opzionale: valuta il tuo RAG system usando il testset generato
    """
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
    )
    
    # Carica il testset
    with open(testset_file, "r", encoding="utf-8") as f:
        testset_data = json.load(f)
    
    # Prepara i dati per la valutazione
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for example in testset_data.get('examples', []):
        question = example.get('question', '')
        
        # Esegui il tuo RAG
        retrieved_docs = retrieve_fn(question)
        answer = llm.generate_final_answer(
            question, 
            retrieved_docs, 
            []
        )
        
        eval_data["question"].append(question)
        eval_data["answer"].append(answer)
        eval_data["contexts"].append([doc.page_content for doc in retrieved_docs])
        eval_data["ground_truth"].append(example.get('ground_truth', ''))
    
    # Valuta
    results = evaluate(
        dataset=eval_data,
        metrics=[
            answer_relevancy,
            faithfulness,
            context_recall,
            context_precision,
        ],
    )
    
    print("\n📈 Risultati valutazione:")
    print(results)
    
    return results


if __name__ == "__main__":
    # Genera il testset
    testset = generate_testset(n_samples=50)
    
    # Opzionale: valuta il sistema
    # results = evaluate_with_testset()