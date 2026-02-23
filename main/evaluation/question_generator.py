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
import llm.search as search
import llm.llm as llm

# Carica le variabili d'ambiente
load_dotenv()

# ===== PROMPT SYSTEM PER FORZARE ITALIANO =====
ITALIAN_SYSTEM_PROMPT = """Sei un assistente esperto che genera domande in ITALIANO per valutare sistemi RAG.

REGOLE CRITICHE:
1. Genera SOLO domande in italiano
2. Usa terminologia tecnica italiana appropriata
3. Le domande devono essere naturali e ben formate in italiano
4. NON mescolare inglese e italiano
5. Mantieni il contesto del software gestionale SAM ERP2
6. Inserisci riferimenti all'azienda cliente nelle domande

Esempio di domande corrette:
- "Come posso configurare un nuovo magazzino in SAM ERP2?"
- "Qual è la procedura per gestire le richieste di implementazione?"
- "Come si risolvono gli errori di validazione nei documenti?"

Genera domande che aiutino a testare la capacità del sistema di recuperare informazioni rilevanti dai documenti."""


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


def retrieve_fn(user_prompt: str):
    """
    Funzione di retrieval che usa la tua logica esistente.
    Ritorna i documenti che il TUO sistema recupererebbe per una domanda.
    """
    # 1 - Decisione strumenti
    tools = llm.decide_tools(user_prompt)

    # 2 - Recupero documenti
    all_documents = []
    if tools["use_semantic"]:
        all_documents += search.semantic_search(user_prompt, 25)

    if tools["use_keyword"]:
        all_documents += search.keyword_search(user_prompt, 25)

    # 3 - Selezione documenti in base alla coerenza
    selected_docs = []
    if tools["use_semantic"] or tools["use_keyword"]:
        if all_documents:
            document_selection = llm.select_documents(user_prompt, all_documents)
            selected_docs = [all_documents[i] for i in document_selection['relevant_docs']]
    
    return selected_docs


def enrich_testset_with_retriever(testset_list, verbose=True):
    """
    Arricchisce il testset eseguendo il TUO retriever su ogni domanda.
    
    Args:
        testset_list: Lista di esempi dal testset Ragas
        verbose: Se True, stampa il progresso
    
    Returns:
        Lista di esempi arricchiti con i documenti recuperati dal tuo sistema
    """
    enriched_examples = []
    
    if verbose:
        print(f"\n🔍 Arricchimento testset con retriever custom...")
        print(f"📊 Domande da processare: {len(testset_list)}")
    
    for i, example in enumerate(testset_list):
        # Estrai la domanda (la chiave può variare in base alla versione di Ragas)
        question = example.get('user_input', example.get('question', ''))
        
        if not question:
            if verbose:
                print(f"  ⚠️ [{i+1}/{len(testset_list)}] Nessuna domanda trovata, skip")
            enriched_examples.append(example)
            continue
        
        if verbose:
            print(f"  [{i+1}/{len(testset_list)}] {question[:60]}...")
        
        # Esegui il TUO retriever sulla domanda
        try:
            retrieved_docs = retrieve_fn(question)
            
            # Estrai contenuti e metadati
            retrieved_contexts = []
            retrieved_metadata = []
            
            for doc in retrieved_docs:
                # Gestisci diversi formati di documento
                if isinstance(doc, dict):
                    content = doc.get('content', '')
                    metadata = {k: v for k, v in doc.items() if k != 'content'}
                elif hasattr(doc, 'page_content'):
                    content = doc.page_content
                    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                else:
                    content = str(doc)
                    metadata = {}
                
                retrieved_contexts.append(content)
                retrieved_metadata.append(metadata)
            
            # Aggiungi i nuovi campi all'esempio
            example['retrieved_contexts'] = retrieved_contexts
            example['retrieved_count'] = len(retrieved_docs)
            example['retrieved_metadata'] = retrieved_metadata
            
            if verbose and len(retrieved_docs) == 0:
                print(f"      ⚠️ Nessun documento recuperato!")
                
        except Exception as e:
            if verbose:
                print(f"      ❌ Errore nel retrieval: {e}")
            
            # In caso di errore, aggiungi campi vuoti
            example['retrieved_contexts'] = []
            example['retrieved_count'] = 0
            example['retrieved_metadata'] = []
        
        enriched_examples.append(example)
    
    if verbose:
        # Stampa statistiche
        total_retrieved = sum(ex.get('retrieved_count', 0) for ex in enriched_examples)
        avg_retrieved = total_retrieved / len(enriched_examples) if enriched_examples else 0
        zero_docs = sum(1 for ex in enriched_examples if ex.get('retrieved_count', 0) == 0)
        
        print(f"\n📊 Statistiche Retrieval:")
        print(f"  • Media documenti recuperati: {avg_retrieved:.2f}")
        print(f"  • Totale documenti recuperati: {total_retrieved}")
        print(f"  • Domande con 0 documenti: {zero_docs}/{len(enriched_examples)}")
    
    return enriched_examples


def create_italian_llm_wrapper(base_llm):
    """
    Crea un wrapper che forza l'LLM a generare in italiano.
    Questo è un workaround per versioni di Ragas senza adapt().
    """
    class ItalianLLMWrapper:
        def __init__(self, llm):
            self.llm = llm
            
        def generate(self, prompts, **kwargs):
            # Aggiungi istruzione italiana a ogni prompt
            italian_prompts = []
            for p in prompts:
                # Se il prompt non menziona già l'italiano, aggiungi l'istruzione
                if "italiano" not in p.lower() and "italian" not in p.lower():
                    italian_prompt = f"{ITALIAN_SYSTEM_PROMPT}\n\n{p}\n\nRICORDA: Genera tutto in ITALIANO."
                    italian_prompts.append(italian_prompt)
                else:
                    italian_prompts.append(p)
            
            return self.llm.generate(italian_prompts, **kwargs)
        
        def __getattr__(self, name):
            # Delega tutti gli altri attributi all'LLM originale
            return getattr(self.llm, name)
    
    return ItalianLLMWrapper(base_llm)


def generate_testset(
    n_samples=30, 
    output_file="testset_ragas.json", 
    enrich_with_retriever=True,
    force_italian=True,
    filter_mixed_language=True
):
    """
    Genera il testset ottimizzato per ITALIANO.
    
    Args:
        n_samples: Numero di domande da generare
        output_file: File di output
        enrich_with_retriever: Se True, esegue il retriever su ogni domanda
        force_italian: Se True, forza la generazione in italiano
        filter_mixed_language: Se True, rimuove domande in inglese o miste
    """
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

    # ===== CONFIGURAZIONE LLM CON FORZATURA ITALIANO =====
    azure_llm = AzureChatOpenAI(
        azure_endpoint="https://cs-test.openai.azure.com",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("LLM_VERSION"),
        deployment_name=os.getenv("LLM_MODEL"),
        temperature=0.4,  # Aumentato leggermente per più varietà
        model_kwargs={
            # Aggiungi context window più ampio se disponibile
            "top_p": 0.95,
        }
    )
    
    # CRITICO: Configura il model con system message in italiano
    if force_italian:
        # Alcuni modelli supportano system message direttamente
        try:
            azure_llm.model_kwargs = azure_llm.model_kwargs or {}
            azure_llm.model_kwargs["messages"] = [
                {"role": "system", "content": ITALIAN_SYSTEM_PROMPT}
            ]
        except:
            pass

    azure_embeddings = AzureOpenAIEmbeddings(
        azure_endpoint="https://cs-test.openai.azure.com",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("EMBEDDING_VERSION_2"),
        deployment=os.getenv("EMBEDDING_MODEL_2"),
    )

    # Inizializzazione del generator di Ragas
    wrapped_llm = LangchainLLMWrapper(azure_llm)
    
    generator = TestsetGenerator(
        llm=wrapped_llm,
        embedding_model=LangchainEmbeddingsWrapper(embeddings=azure_embeddings),
    )
    
    try:
        num_docs = min(50, len(llama_docs))
        print(f"\n🎯 Generazione di {n_samples} domande in ITALIANO da {num_docs} documenti...")
        print("⚠️  NOTA: Le prime domande potrebbero essere in inglese, continua la generazione...")

        testset = generator.generate_with_llamaindex_docs(
            documents=llama_docs[:num_docs],
            testset_size=n_samples * 2 if filter_mixed_language else n_samples,  # Genera il doppio se filtriamo
            raise_exceptions=True,
        )
        print(f"✅ Testset base generato! Numero domande: {len(testset)}")
        
        testset_list = testset.to_list()
        
        # ===== FILTRAGGIO DOMANDE IN INGLESE/MISTE =====
        if filter_mixed_language:
            print(f"\n🔍 Filtro domande in inglese o miste...")
            original_count = len(testset_list)
            
            italian_questions = []
            for ex in testset_list:
                question = ex.get('user_input', ex.get('question', ''))
                
                # Controlli semplici per identificare italiano
                # 1. Deve contenere caratteri accentati tipici dell'italiano
                # 2. Non deve contenere pattern tipici dell'inglese
                has_italian_chars = any(c in question for c in ['à', 'è', 'é', 'ì', 'ò', 'ù'])
                has_english_words = any(word in question.lower() for word in [
                    ' the ', ' is ', ' are ', ' was ', ' were ', ' have ', ' has ',
                    'what', 'how', 'when', 'where', 'which', 'who'
                ])
                
                # Parole italiane comuni che indicano una domanda italiana
                has_italian_question_words = any(word in question.lower() for word in [
                    'come', 'cosa', 'quando', 'dove', 'quale', 'quali', 'chi', 'perché',
                    'può', 'posso', 'possono', 'deve', 'devo', 'devono'
                ])
                
                # Mantieni se sembra italiano
                if (has_italian_chars or has_italian_question_words) and not has_english_words:
                    italian_questions.append(ex)
                
                # Interrompi quando hai abbastanza domande
                if len(italian_questions) >= n_samples:
                    break
            
            testset_list = italian_questions[:n_samples]
            filtered_count = original_count - len(testset_list)
            print(f"✅ Filtrate {filtered_count} domande non italiane")
            print(f"📊 Domande italiane valide: {len(testset_list)}")
        
        # ===== ARRICCHIMENTO CON RETRIEVER =====
        if enrich_with_retriever:
            print("\n" + "="*70)
            testset_list = enrich_testset_with_retriever(testset_list, verbose=True)
            print("="*70)
            print("✅ Testset arricchito con retriever custom")
        
        # Salva il testset
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(testset_list, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Testset salvato in: {output_file}")
        
        # Mostra esempi di output
        if testset_list and len(testset_list) > 0:
            print("\n💡 Esempi di domande generate in ITALIANO:")
            for i, example in enumerate(testset_list[:3], 1):
                question = example.get('user_input', example.get('question', 'N/A'))
                print(f"\n  {i}. {question}")
                if enrich_with_retriever:
                    print(f"     → Documenti recuperati: {example.get('retrieved_count', 0)}")
        
        return testset_list
        
    except Exception as e:
        print(f"❌ Errore: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def post_process_to_italian(testset_file="testset_ragas.json"):
    """
    Post-processing per tradurre/filtrare domande non italiane.
    Usa questa funzione se il testset generato contiene troppe domande in inglese.
    """
    print("\n🔄 Post-processing per garantire italiano...")
    
    with open(testset_file, "r", encoding="utf-8") as f:
        testset = json.load(f)
    
    # Identifica domande non italiane
    non_italian = []
    for i, ex in enumerate(testset):
        question = ex.get('user_input', ex.get('question', ''))
        
        # Controlla se è in inglese
        english_indicators = ['what', 'how', 'when', 'where', 'the', 'is', 'are']
        if any(word in question.lower().split() for word in english_indicators):
            non_italian.append((i, question))
    
    if non_italian:
        print(f"\n⚠️ Trovate {len(non_italian)} domande probabilmente in inglese:")
        for idx, q in non_italian[:5]:  # Mostra prime 5
            print(f"  {idx+1}. {q[:60]}...")
        
        print("\n💡 Suggerimenti:")
        print("  1. Aumenta 'testset_size' e filtra con filter_mixed_language=True")
        print("  2. Usa un modello più grande (es. GPT-4) per migliore comprensione")
        print("  3. Aggiungi esempi italiani nei documenti di input")
    else:
        print("✅ Tutte le domande sembrano essere in italiano!")
    
    return len(non_italian)


if __name__ == "__main__":
    # ===== CONFIGURAZIONE OTTIMIZZATA PER ITALIANO =====
    N_SAMPLES = 30
    OUTPUT_FILE = "testset_ragas_italiano.json"
    
    # Genera testset CON filtro italiano
    testset = generate_testset(
        n_samples=N_SAMPLES,
        output_file=OUTPUT_FILE,
        enrich_with_retriever=True,
        force_italian=True,          # Forza generazione in italiano
        filter_mixed_language=True    # Filtra domande miste/inglesi
    )
    
    if testset:
        print("\n✨ Testset generato con successo!")
        print("\n📋 Struttura dati:")
        print("  • user_input: La domanda in ITALIANO")
        print("  • reference: Risposta attesa in ITALIANO")
        print("  • reference_contexts: Documenti usati da Ragas")
        print("  • retrieved_contexts: Documenti trovati dal TUO retriever ⭐")
        print("  • retrieved_count: Numero documenti recuperati ⭐")
        print("  • retrieved_metadata: Metadati dei documenti (RI, cliente, ecc.) ⭐")
        
        # Verifica qualità italiano
        print("\n🔍 Verifica qualità italiano nel testset...")
        non_italian_count = post_process_to_italian(OUTPUT_FILE)
        
        if non_italian_count == 0:
            print("\n🎉 Ottimo! Tutte le domande sono in italiano!")
        else:
            print(f"\n⚠️ Ci sono {non_italian_count} domande da rivedere.")
            print("💡 Considera di rigenerare con testset_size più alto e filter_mixed_language=True")
        
        print("\nPuoi ora usarlo per valutare il tuo sistema RAG.")
    else:
        print("\n⚠️ Generazione testset fallita. Controlla gli errori sopra.")