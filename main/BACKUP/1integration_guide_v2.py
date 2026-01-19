"""
GUIDA ALL'INTEGRAZIONE - Pipeline di Valutazione RAG
Versione 2.0 - Approccio funzionale con LLM-as-judge

Questa guida mostra come integrare la pipeline con il tuo sistema RAG esistente
e come gestire test standalone con diverse configurazioni.
"""

import time
from typing import List
from evaluation_pipeline import (
    load_gold_dataset,
    load_openai_client,
    run_full_evaluation,
    save_test_result,
    load_all_test_results,
    compare_test_results,
    print_test_result,
    EvaluationQuery,
    RetrievalResult,
    GenerationResult
)


# ==================== INTEGRAZIONE CON IL TUO SISTEMA RAG ====================

def run_retrieval_on_queries(
    queries: List[EvaluationQuery],
    your_rag_system  # Sostituisci con il tipo del tuo sistema
) -> List[RetrievalResult]:
    """
    Esegue il retrieval per tutte le query del dataset usando il tuo sistema RAG.
    
    ⚠️ QUESTA FUNZIONE VA PERSONALIZZATA con le chiamate al tuo sistema.
    
    Args:
        queries: Lista delle query da valutare
        your_rag_system: Istanza del tuo sistema RAG
        
    Returns:
        Lista di RetrievalResult
    """
    retrieval_results = []
    
    for query in queries:
        start_time = time.time()
        
        # ========================================
        # ⚠️ MODIFICA QUI - Chiama il tuo retriever
        # ========================================
        
        # Esempio di chiamata (DA ADATTARE):
        # retrieved_docs = your_rag_system.retrieve(
        #     query=query.query_text,
        #     top_k=10
        # )
        # 
        # retrieved_chunk_ids = [doc.chunk_id for doc in retrieved_docs]
        # retrieved_chunk_texts = [doc.text for doc in retrieved_docs]
        
        # PLACEHOLDER (da rimuovere):
        retrieved_chunk_ids = []  # Inserisci gli ID effettivi
        retrieved_chunk_texts = []  # Inserisci i testi effettivi
        
        # ========================================
        
        retrieval_time = time.time() - start_time
        
        result = RetrievalResult(
            query_id=query.query_id,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_chunk_texts=retrieved_chunk_texts,
            retrieval_time=retrieval_time
        )
        retrieval_results.append(result)
    
    return retrieval_results


def run_generation_on_queries(
    queries: List[EvaluationQuery],
    retrieval_results: List[RetrievalResult],
    your_rag_system  # Sostituisci con il tipo del tuo sistema
) -> List[GenerationResult]:
    """
    Esegue la generation per tutte le query usando i chunk recuperati.
    
    ⚠️ QUESTA FUNZIONE VA PERSONALIZZATA con le chiamate al tuo sistema.
    
    Args:
        queries: Lista delle query da valutare
        retrieval_results: Risultati del retrieval precedente
        your_rag_system: Istanza del tuo sistema RAG
        
    Returns:
        Lista di GenerationResult
    """
    # Mappa query_id -> retrieved chunks
    retrieval_map = {
        r.query_id: (r.retrieved_chunk_ids, r.retrieved_chunk_texts) 
        for r in retrieval_results
    }
    
    generation_results = []
    
    for query in queries:
        chunk_ids, chunk_texts = retrieval_map.get(query.query_id, ([], []))
        
        start_time = time.time()
        
        # ========================================
        # ⚠️ MODIFICA QUI - Chiama il tuo generator
        # ========================================
        
        # Esempio di chiamata (DA ADATTARE):
        # answer = your_rag_system.generate(
        #     query=query.query_text,
        #     context_chunks=chunk_texts
        # )
        
        # PLACEHOLDER (da rimuovere):
        answer = ""  # Inserisci la risposta effettiva
        
        # ========================================
        
        generation_time = time.time() - start_time
        
        result = GenerationResult(
            query_id=query.query_id,
            generated_answer=answer,
            context_chunks=chunk_texts,
            generation_time=generation_time
        )
        generation_results.append(result)
    
    return generation_results


# ==================== WORKFLOW PER TEST SINGOLO ====================

def run_single_test(
    gold_dataset_path: str,
    your_rag_system,
    configuration: dict,
    results_dir: str = "evaluation_results"
):
    """
    Esegue un singolo test completo di valutazione.
    
    Questo è il workflow principale da usare per ogni configurazione che vuoi testare.
    
    Args:
        gold_dataset_path: Percorso al dataset GOLD
        your_rag_system: Istanza del tuo sistema RAG
        configuration: Dizionario con la configurazione del test
        results_dir: Directory dove salvare i risultati
    """
    print("\n" + "="*70)
    print(f"TEST: {configuration.get('name', 'Unnamed')}")
    print("="*70)
    
    # 1. Carica dataset GOLD
    print("\n[1/4] Caricamento dataset GOLD...")
    gold_queries = load_gold_dataset(gold_dataset_path)
    
    # 2. Esegui retrieval
    print("\n[2/4] Esecuzione retrieval...")
    retrieval_results = run_retrieval_on_queries(gold_queries, your_rag_system)
    
    # 3. Esegui generation
    print("\n[3/4] Esecuzione generation...")
    generation_results = run_generation_on_queries(
        gold_queries, retrieval_results, your_rag_system
    )
    
    # 4. Valuta e salva
    print("\n[4/4] Valutazione e salvataggio...")
    client = load_openai_client()
    
    test_result = run_full_evaluation(
        gold_queries=gold_queries,
        retrieval_results=retrieval_results,
        generation_results=generation_results,
        configuration=configuration,
        client=client,
        k_values=[1, 3, 5, 10],
        llm_model=configuration.get("llm_model", "gpt-4o-mini"),
        embedding_model=configuration.get("embedding_model", "text-embedding-3-large")
    )
    
    print_test_result(test_result)
    save_test_result(test_result, results_dir)
    
    return test_result


# ==================== ESEMPI DI WORKFLOW ====================

def esempio_test_baseline():
    """
    Esempio: Esegui un test baseline.
    """
    # 1. Inizializza il tuo sistema RAG con configurazione baseline
    # your_rag = YourRAGSystem(
    #     embedding_model="text-embedding-3-large",
    #     llm_model="gpt-4o-mini",
    #     chunk_size=512,
    #     chunk_overlap=50,
    #     top_k=5
    # )
    
    # 2. Definisci configurazione del test
    configuration = {
        "name": "Baseline - gpt-4o-mini, chunk 512",
        "embedding_model": "text-embedding-3-large",
        "llm_model": "gpt-4o-mini",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "top_k": 5,
        "note": "Configurazione di riferimento per confronti futuri"
    }
    
    # 3. Esegui test
    # test_result = run_single_test(
    #     gold_dataset_path="path/to/gold_dataset.json",
    #     your_rag_system=your_rag,
    #     configuration=configuration
    # )
    
    print("✓ Test baseline completato e salvato!")


def esempio_test_llm_migliore():
    """
    Esempio: Test con LLM più potente (gpt-4o).
    
    NOTA: Cambiare solo il modello LLM NON richiede di rifare il chunking!
    """
    # 1. Inizializza il tuo sistema RAG con LLM diverso
    # your_rag = YourRAGSystem(
    #     embedding_model="text-embedding-3-large",
    #     llm_model="gpt-4o",  # ← Cambiato solo questo
    #     chunk_size=512,
    #     chunk_overlap=50,
    #     top_k=5
    # )
    
    # 2. Definisci configurazione
    configuration = {
        "name": "Test LLM - gpt-4o, chunk 512",
        "embedding_model": "text-embedding-3-large",
        "llm_model": "gpt-4o",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "top_k": 5,
        "note": "Test con modello più potente"
    }
    
    # 3. Esegui test
    # test_result = run_single_test(
    #     gold_dataset_path="path/to/gold_dataset.json",
    #     your_rag_system=your_rag,
    #     configuration=configuration
    # )
    
    print("✓ Test LLM completato e salvato!")


def esempio_test_embedding_diverso():
    """
    Esempio: Test con modello di embedding diverso.
    
    ⚠️ ATTENZIONE: Cambiare embedding model RICHIEDE di rifare il vector store!
    Questo va fatto manualmente prima di eseguire questo test.
    """
    # STEP PRELIMINARE (da fare una volta manualmente):
    # 1. Ricostruisci il vector store con text-embedding-ada-002
    # 2. Verifica che il sistema RAG usi il nuovo vector store
    
    # Poi esegui il test:
    
    # your_rag = YourRAGSystem(
    #     embedding_model="text-embedding-ada-002",  # ← Embedding diverso
    #     llm_model="gpt-4o-mini",
    #     chunk_size=512,
    #     chunk_overlap=50,
    #     top_k=5
    # )
    
    configuration = {
        "name": "Test Embedding - ada-002, chunk 512",
        "embedding_model": "text-embedding-ada-002",
        "llm_model": "gpt-4o-mini",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "top_k": 5,
        "note": "Vector store ricostruito con ada-002"
    }
    
    # test_result = run_single_test(
    #     gold_dataset_path="path/to/gold_dataset.json",
    #     your_rag_system=your_rag,
    #     configuration=configuration
    # )
    
    print("✓ Test embedding completato e salvato!")


def esempio_test_top_k_diverso():
    """
    Esempio: Test con parametro top_k diverso.
    
    NOTA: Cambiare solo top_k NON richiede di rifare il chunking!
    """
    # your_rag = YourRAGSystem(
    #     embedding_model="text-embedding-3-large",
    #     llm_model="gpt-4o-mini",
    #     chunk_size=512,
    #     chunk_overlap=50,
    #     top_k=10  # ← Cambiato solo questo
    # )
    
    configuration = {
        "name": "Test top_k - k=10, chunk 512",
        "embedding_model": "text-embedding-3-large",
        "llm_model": "gpt-4o-mini",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "top_k": 10,
        "note": "Test con più chunk recuperati"
    }
    
    # test_result = run_single_test(
    #     gold_dataset_path="path/to/gold_dataset.json",
    #     your_rag_system=your_rag,
    #     configuration=configuration
    # )
    
    print("✓ Test top_k completato e salvato!")


# ==================== CONFRONTO RISULTATI ====================

def confronta_tutti_i_risultati():
    """
    Carica e confronta tutti i risultati salvati.
    
    Usa questo dopo aver eseguito più test con diverse configurazioni.
    """
    print("\n" + "="*70)
    print("CARICAMENTO E CONFRONTO RISULTATI")
    print("="*70)
    
    # Carica tutti i risultati salvati
    results = load_all_test_results("evaluation_results")
    
    if not results:
        print("\n⚠️  Nessun risultato trovato in evaluation_results/")
        print("Esegui prima alcuni test con run_single_test()")
        return
    
    print(f"\n✓ Caricati {len(results)} test")
    
    # Ordina per timestamp
    results.sort(key=lambda r: r.timestamp)
    
    # Mostra confronto
    compare_test_results(results)
    
    # Identifica configurazione migliore
    print("\n🏆 ANALISI BEST PERFORMANCE:")
    
    best_precision = max(results, key=lambda r: r.retrieval_metrics.get('avg_precision_at_5', 0))
    print(f"\nMiglior Precision@5: {best_precision.configuration.get('name')}")
    print(f"  → {best_precision.retrieval_metrics.get('avg_precision_at_5', 0):.4f}")
    
    best_faithfulness = max(results, key=lambda r: r.generation_metrics.get('avg_faithfulness', 0))
    print(f"\nMiglior Faithfulness: {best_faithfulness.configuration.get('name')}")
    print(f"  → {best_faithfulness.generation_metrics.get('avg_faithfulness', 0):.4f}")
    
    best_relevancy = max(results, key=lambda r: r.generation_metrics.get('avg_answer_relevancy', 0))
    print(f"\nMiglior Answer Relevancy: {best_relevancy.configuration.get('name')}")
    print(f"  → {best_relevancy.generation_metrics.get('avg_answer_relevancy', 0):.4f}")


# ==================== WORKFLOW COMPLETO PER LA TESI ====================

def workflow_completo_tesi():
    """
    Workflow completo suggerito per gli esperimenti di tesi.
    
    Questo mostra l'ordine consigliato dei test per analizzare sistematicamente
    l'impatto di ogni parametro.
    """
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║           WORKFLOW COMPLETO ESPERIMENTI TESI                      ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    FASE 1: BASELINE
    ================
    • Test 1: Configurazione baseline di riferimento
      - embedding: text-embedding-3-large
      - llm: gpt-4o-mini
      - chunk_size: 512
      - top_k: 5
    
    FASE 2: VARIAZIONE LLM (non richiede rebuild)
    ==============================================
    • Test 2: Stesso setup, LLM → gpt-4o
      → Confronto: qualità generazione con modello più potente
    
    FASE 3: VARIAZIONE TOP_K (non richiede rebuild)
    ================================================
    • Test 3: Baseline, top_k → 3
    • Test 4: Baseline, top_k → 10
      → Confronto: impatto numero chunk recuperati
    
    FASE 4: VARIAZIONE EMBEDDING (richiede rebuild vector store)
    =============================================================
    ⚠️  Per ognuno di questi, ricostruire il vector store prima del test:
    
    • Test 5: embedding → text-embedding-ada-002
      → Confronto: qualità retrieval con modello meno costoso
    
    FASE 5: VARIAZIONE CHUNK SIZE (richiede rebuild vector store)
    ==============================================================
    ⚠️  Per ognuno di questi, riprocessare documenti e ricostruire vector store:
    
    • Test 6: chunk_size → 256, chunk_overlap → 25
    • Test 7: chunk_size → 1024, chunk_overlap → 100
    • Test 8: chunk_size → 2048, chunk_overlap → 200
      → Confronto: impatto granularità chunk
    
    FASE 6: ANALISI FINALE
    =======================
    • Esegui confronta_tutti_i_risultati()
    • Identifica trade-off (es: precision vs recall, costo vs qualità)
    • Documenta configurazione ottimale per il tuo caso d'uso
    • Genera grafici per la tesi
    
    ════════════════════════════════════════════════════════════════════
    
    SUGGERIMENTO: Esegui i test nell'ordine sopra indicato per minimizzare
    il numero di volte che devi ricostruire il vector store.
    """)


# ==================== CHECKLIST ====================

def stampa_checklist_integrazione():
    """
    Checklist per l'integrazione completa.
    """
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                 CHECKLIST INTEGRAZIONE                            ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    SETUP INIZIALE:
    ☐ 1. Aggiungi dipendenze al pyproject.toml:
         openai = "^1.0.0"
         numpy = "^1.24.0"
    
    ☐ 2. Esegui: uv sync
    
    ☐ 3. Configura variabile ambiente OPENAI_API_KEY
    
    PREPARAZIONE DATASET:
    ☐ 4. Crea dataset GOLD in JSON con:
         - query_id, query_text, relevant_chunk_ids
         - (opzionale) reference_answer
    
    ☐ 5. Verifica che gli ID chunk corrispondano a quelli nel tuo sistema
    
    INTEGRAZIONE CODICE:
    ☐ 6. Personalizza run_retrieval_on_queries():
         - Implementa chiamata al tuo retriever
         - Assicurati di restituire sia ID che testi dei chunk
    
    ☐ 7. Personalizza run_generation_on_queries():
         - Implementa chiamata al tuo generator
         - Passa i chunk come contesto
    
    ESECUZIONE TEST:
    ☐ 8. Esegui test baseline con run_single_test()
    
    ☐ 9. Verifica che i risultati siano salvati in evaluation_results/
    
    ☐ 10. Esegui test con variazioni di configurazione
    
    ANALISI:
    ☐ 11. Usa confronta_tutti_i_risultati() per confronto
    
    ☐ 12. Documenta findings per la tesi
    
    ════════════════════════════════════════════════════════════════════
    """)


if __name__ == "__main__":
    print("="*70)
    print("GUIDA INTEGRAZIONE - Pipeline Valutazione RAG")
    print("="*70)
    
    stampa_checklist_integrazione()
    workflow_completo_tesi()
    
    print("\n\nPer esempi di codice, vedi le funzioni esempio_*() in questo file.")
    print("Per confrontare risultati esistenti, esegui confronta_tutti_i_risultati()")
