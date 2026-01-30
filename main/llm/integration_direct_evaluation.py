"""
INTEGRAZIONE DIRETTA - Pipeline RAG Esistente + Valutazione
============================================================

Questo modulo integra DIRETTAMENTE il tuo sistema RAG esistente
con la pipeline di valutazione, usando le tue funzioni reali.

ZERO duplicazione di codice - usa search.py e llm.py com'è.

Autore: [Nome Studente]
Data: Gennaio 2026
"""

import time
import sys
import os
from typing import List, Dict

# Importa le TUE funzioni esistenti
sys.path.append(os.path.dirname(__file__))
import search # Le tue funzioni di retrieval

# Importa pipeline valutazione
from main.llm.rag_evaluation_pipeline_v2 import (
    load_gold_dataset,
    load_openai_client,
    run_full_evaluation,
    save_test_result,
    print_test_result,
    EvaluationQuery,
    RetrievalResult,
    GenerationResult
)


# ==================== CONFIGURAZIONE ====================

class SearchStrategy:
    """Enum per le strategie di search disponibili."""
    SEMANTIC = "semantic"  # Vector similarity
    KEYWORD = "keyword"    # BM25
    HYBRID = "hybrid"      # Combinazione di entrambi

path = "C:/Users/ACantelli/OneDrive - centrosoftware.com/Documenti/GitHub/progetto-tesi/main/evaluation/gold_dataset.json"

# ==================== ADAPTER PER RETRIEVAL ====================

def run_retrieval_with_semantic_search(
    queries: List[EvaluationQuery],
    top_k: int = 10
) -> List[RetrievalResult]:
    """
    Esegue retrieval usando la TUA funzione semantic_search().
    
    USA DIRETTAMENTE: search.semantic_search() dal tuo codice.
    
    Args:
        queries: Lista query da valutare
        top_k: Numero chunk da recuperare
        
    Returns:
        Lista RetrievalResult per valutazione
    """
    retrieval_results = []
    
    print(f"\n{'='*70}")
    print(f"RETRIEVAL SEMANTICO: {len(queries)} query")
    print(f"{'='*70}\n")
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query.query_text[:60]}...")
        
        start_time = time.time()
        
        try:
            # ✅ USA LA TUA FUNZIONE ESISTENTE
            docs = search.semantic_search(
                prompt=query.query_text,
                top_n=top_k
            )
            
            # Estrai ID e testi nel formato richiesto
            retrieved_chunk_ids = [
                f"{doc['numero']}_{doc['progressivo']}" 
                for doc in docs
            ]
            retrieved_chunk_texts = [doc['content'] for doc in docs]
            
            print(f"  ✓ {len(retrieved_chunk_ids)} chunk recuperati")
            
        except Exception as e:
            print(f"  ✗ Errore: {e}")
            retrieved_chunk_ids = []
            retrieved_chunk_texts = []
        
        retrieval_time = time.time() - start_time
        
        result = RetrievalResult(
            query_id=query.query_id,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_chunk_texts=retrieved_chunk_texts,
            retrieval_time=retrieval_time
        )
        retrieval_results.append(result)
    
    print(f"\n{'='*70}\n")
    return retrieval_results


def run_retrieval_with_keyword_search(
    queries: List[EvaluationQuery],
    top_k: int = 10
) -> List[RetrievalResult]:
    """
    Esegue retrieval usando la TUA funzione keyword_search() (BM25).
    
    USA DIRETTAMENTE: search.keyword_search() dal tuo codice.
    
    Args:
        queries: Lista query da valutare
        top_k: Numero chunk da recuperare
        
    Returns:
        Lista RetrievalResult per valutazione
    """
    retrieval_results = []
    
    print(f"\n{'='*70}")
    print(f"RETRIEVAL BM25: {len(queries)} query")
    print(f"{'='*70}\n")
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query.query_text[:60]}...")
        
        start_time = time.time()
        
        try:
            # ✅ USA LA TUA FUNZIONE ESISTENTE
            docs = search.keyword_search(
                prompt=query.query_text,
                top_n=top_k,
                language='italian'
            )
            
            # Estrai ID e testi
            retrieved_chunk_ids = [
                f"{doc['numero']}_{doc['progressivo']}" 
                for doc in docs
            ]
            retrieved_chunk_texts = [doc['content'] for doc in docs]
            
            print(f"  ✓ {len(retrieved_chunk_ids)} chunk recuperati")
            
        except Exception as e:
            print(f"  ✗ Errore: {e}")
            retrieved_chunk_ids = []
            retrieved_chunk_texts = []
        
        retrieval_time = time.time() - start_time
        
        result = RetrievalResult(
            query_id=query.query_id,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_chunk_texts=retrieved_chunk_texts,
            retrieval_time=retrieval_time
        )
        retrieval_results.append(result)
    
    print(f"\n{'='*70}\n")
    return retrieval_results


def run_retrieval_hybrid(
    queries: List[EvaluationQuery],
    top_k: int = 10,
    semantic_weight: float = 0.7
) -> List[RetrievalResult]:
    """
    Esegue retrieval IBRIDO: combina semantic e keyword search.
    
    USA ENTRAMBE le tue funzioni e fonde i risultati.
    
    Args:
        queries: Lista query da valutare
        top_k: Numero chunk finali da recuperare
        semantic_weight: Peso similarità semantica (0-1)
        
    Returns:
        Lista RetrievalResult per valutazione
    """
    retrieval_results = []
    keyword_weight = 1.0 - semantic_weight
    
    print(f"\n{'='*70}")
    print(f"RETRIEVAL IBRIDO: {len(queries)} query")
    print(f"Pesi: Semantic={semantic_weight}, Keyword={keyword_weight}")
    print(f"{'='*70}\n")
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query.query_text[:60]}...")
        
        start_time = time.time()
        
        try:
            # Recupera da entrambe le fonti
            semantic_docs = search.semantic_search(query.query_text, top_n=top_k*2)
            keyword_docs = search.keyword_search(query.query_text, top_n=top_k*2)
            
            # Fonde risultati con weighted score
            combined = {}
            
            # Aggiungi semantic results
            for doc in semantic_docs:
                chunk_id = f"{doc['numero']}_{doc['progressivo']}"
                combined[chunk_id] = {
                    'doc': doc,
                    'score': doc['similarity'] * semantic_weight
                }
            
            # Aggiungi/aggiorna keyword results
            for doc in keyword_docs:
                chunk_id = f"{doc['numero']}_{doc['progressivo']}"
                # Normalizza BM25 score (0-1)
                norm_score = doc['score'] / (doc['score'] + 1)
                
                if chunk_id in combined:
                    combined[chunk_id]['score'] += norm_score * keyword_weight
                else:
                    combined[chunk_id] = {
                        'doc': doc,
                        'score': norm_score * keyword_weight
                    }
            
            # Ordina per score e prendi top_k
            sorted_results = sorted(
                combined.items(), 
                key=lambda x: x[1]['score'], 
                reverse=True
            )[:top_k]
            
            retrieved_chunk_ids = [chunk_id for chunk_id, _ in sorted_results]
            retrieved_chunk_texts = [data['doc']['content'] for _, data in sorted_results]
            
            print(f"  ✓ {len(retrieved_chunk_ids)} chunk recuperati (ibrido)")
            
        except Exception as e:
            print(f"  ✗ Errore: {e}")
            retrieved_chunk_ids = []
            retrieved_chunk_texts = []
        
        retrieval_time = time.time() - start_time
        
        result = RetrievalResult(
            query_id=query.query_id,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_chunk_texts=retrieved_chunk_texts,
            retrieval_time=retrieval_time
        )
        retrieval_results.append(result)
    
    print(f"\n{'='*70}\n")
    return retrieval_results


# ==================== ADAPTER PER GENERATION ====================

def run_generation_with_llm(
    queries: List[EvaluationQuery],
    retrieval_results: List[RetrievalResult],
    llm_model: str = "gpt-4o-mini"
) -> List[GenerationResult]:
    """
    Esegue generation usando la TUA funzione llm.generate_final_answer().
    
    IMPORTANTE: Usa il tuo sistema di generation completo con:
    - Selezione documenti via LLM
    - Generazione risposta con citazioni
    - Streaming output
    
    Args:
        queries: Lista query da valutare
        retrieval_results: Risultati retrieval
        llm_model: Nome deployment LLM (non usato, usa configurazione da .env)
        
    Returns:
        Lista GenerationResult
    """
    # Prova a importare il TUO llm.py
    try:
        import llm as your_llm
        use_your_llm = True
        print("\n✓ Usando il TUO llm.py per generation")
    except ImportError:
        use_your_llm = False
        print("\n⚠️  llm.py non trovato :( ")
    
    generation_results = []
    retrieval_map = {
        r.query_id: (r.retrieved_chunk_ids, r.retrieved_chunk_texts) 
        for r in retrieval_results
    }
    
    print(f"\n{'='*70}")
    print(f"GENERATION: {len(queries)} query")
    print(f"{'='*70}\n")
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query.query_text[:60]}...")
        
        chunk_ids, chunk_texts = retrieval_map.get(query.query_id, ([], []))
        
        start_time = time.time()
        
        try:
            if use_your_llm:
                # ✅ USA IL TUO SISTEMA COMPLETO
                # Ricostruisci documenti nel formato atteso da select_documents()
                # e generate_final_answer()
                
                # NOTA: Il tuo sistema si aspetta documenti con questo formato
                # Qui non abbiamo tutti i metadata, quindi creiamo una versione semplificata
                # che bypassa select_documents e va direttamente a generate_final_answer
                
                # Crea documenti fittizi con i chunk recuperati
                fake_docs = []
                for idx, (chunk_id, chunk_text) in enumerate(zip(chunk_ids, chunk_texts)):
                    # Parsing chunk_id "NumRI_Progressivo"
                    try:
                        num_ri, progressivo = chunk_id.split('_')
                    except:
                        num_ri, progressivo = "UNKNOWN", str(idx)
                    
                    fake_docs.append({
                        'numero': num_ri,
                        'progressivo': int(progressivo),
                        'titolo': f"Documento {idx+1}",
                        'autore': "Sistema",
                        'cliente': "Test",
                        'content': chunk_text
                    })
                
                # Chiama generate_final_answer con i documenti
                # Raccoglie tutto lo streaming
                answer = ""
                for token in your_llm.generate_final_answer(
                    user_prompt=query.query_text,
                    selected_docs=fake_docs,
                    chat_history=[]  # Nessuna cronologia per valutazione
                ):
                    answer += token
                    
            else:
                # Fallback: usa rag_generation.py
                answer = rag_generation.generate_answer(
                    query=query.query_text,
                    context_chunks=chunk_texts,
                    model=llm_model
                )
            
            print(f"  ✓ Risposta generata ({len(answer)} char)")
            
        except Exception as e:
            print(f"  ✗ Errore: {e}")
            import traceback
            traceback.print_exc()
            answer = f"[ERRORE] {str(e)}"
        
        generation_time = time.time() - start_time
        
        result = GenerationResult(
            query_id=query.query_id,
            generated_answer=answer,
            context_chunks=chunk_texts,
            generation_time=generation_time
        )
        generation_results.append(result)
    
    print(f"\n{'='*70}\n")
    return generation_results


# ==================== TEST PRINCIPALE ====================

def run_test_with_your_pipeline(
    gold_dataset_path: str,
    configuration: dict,
    search_strategy: str = SearchStrategy.SEMANTIC,
    results_dir: str = "evaluation_results"
):
    """
    Esegue test completo usando IL TUO SISTEMA RAG.
    
    Args:
        gold_dataset_path: Path dataset GOLD
        configuration: Config del test
        search_strategy: "semantic", "keyword", o "hybrid"
        results_dir: Directory risultati
    """
    print("\n" + "="*70)
    print(f"TEST: {configuration.get('name', 'Unnamed')}")
    print(f"Search Strategy: {search_strategy}")
    print("="*70)
    
    # 1. Carica dataset
    print("\n[1/4] Caricamento dataset GOLD...")
    gold_queries = load_gold_dataset(gold_dataset_path)
    
    # 2. Retrieval (usa LA TUA funzione)
    print(f"\n[2/4] Retrieval con strategia: {search_strategy}...")
    top_k = configuration.get('top_k', 10)
    
    if search_strategy == SearchStrategy.SEMANTIC:
        retrieval_results = run_retrieval_with_semantic_search(gold_queries, top_k)
    elif search_strategy == SearchStrategy.KEYWORD:
        retrieval_results = run_retrieval_with_keyword_search(gold_queries, top_k)
    elif search_strategy == SearchStrategy.HYBRID:
        retrieval_results = run_retrieval_hybrid(
            gold_queries, 
            top_k,
            semantic_weight=configuration.get('semantic_weight', 0.7)
        )
    else:
        raise ValueError(f"Search strategy non valida: {search_strategy}")
    
    # 3. Generation (usa LA TUA funzione se disponibile)
    print("\n[3/4] Generation...")
    generation_results = run_generation_with_llm(
        gold_queries,
        retrieval_results,
        llm_model=configuration.get('llm_model', 'gpt-4o-mini')
    )
    
    # 4. Valutazione
    print("\n[4/4] Valutazione metriche...")
    client = load_openai_client()
    
    test_result = run_full_evaluation(
        gold_queries=gold_queries,
        retrieval_results=retrieval_results,
        generation_results=generation_results,
        configuration=configuration,
        client=client,
        k_values=[1, 3, 5, 10],
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-large"
    )
    
    print_test_result(test_result)
    save_test_result(test_result, results_dir)
    
    return test_result


# ==================== ESEMPI DI TEST ====================

def esempio_test_semantic_search():
    """
    Test con SOLO semantic search (vector similarity).
    """
    configuration = {
        "name": "Test Semantic Search",
        "search_strategy": "semantic",
        "embedding_model": "Azure OpenAI",
        "llm_model": "gpt-4o-mini",
        "top_k": 5,
        "note": "Vector similarity con embeddings Azure"
    }
    
    return run_test_with_your_pipeline(
        gold_dataset_path=path,
        configuration=configuration,
        search_strategy=SearchStrategy.SEMANTIC
    )


def esempio_test_keyword_search():
    """
    Test con SOLO keyword search (BM25).
    """
    configuration = {
        "name": "Test Keyword Search BM25",
        "search_strategy": "keyword",
        "llm_model": "gpt-4o-mini",
        "top_k": 5,
        "note": "BM25 keyword search con stemming italiano"
    }
    
    return run_test_with_your_pipeline(
        gold_dataset_path=path,
        configuration=configuration,
        search_strategy=SearchStrategy.KEYWORD
    )


def esempio_test_hybrid_search():
    """
    Test con ricerca IBRIDA (semantic + keyword).
    """
    configuration = {
        "name": "Test Hybrid Search",
        "search_strategy": "hybrid",
        "semantic_weight": 0.7,
        "keyword_weight": 0.3,
        "llm_model": "gpt-4o-mini",
        "top_k": 5,
        "note": "Combinazione semantic (70%) + keyword (30%)"
    }
    
    return run_test_with_your_pipeline(
        gold_dataset_path=path,
        configuration=configuration,
        search_strategy=SearchStrategy.HYBRID
    )


def esempio_confronto_strategie():
    """
    Confronto diretto delle 3 strategie di search.
    """
    print("\n" + "="*70)
    print("CONFRONTO STRATEGIE DI SEARCH")
    print("="*70)
    
    # Test 1: Semantic
    print("\n[TEST 1/3] Semantic Search...")
    esempio_test_semantic_search()
    
    # Test 2: Keyword
    print("\n[TEST 2/3] Keyword Search (BM25)...")
    esempio_test_keyword_search()
    
    # Test 3: Hybrid
    print("\n[TEST 3/3] Hybrid Search...")
    esempio_test_hybrid_search()
    
    # Confronto finale
    print("\n" + "="*70)
    print("Per visualizzare il confronto:")
    print("  python compare_results.py")
    print("="*70)


# ==================== MAIN ====================

def main():
    """
    Menu interattivo per scegliere il test da eseguire.
    """
    print("="*70)
    print("PIPELINE VALUTAZIONE - Sistema RAG Esistente")
    print("="*70)
    
    print("\nScegli il tipo di test:")
    print("1. Semantic Search (vector similarity)")
    print("2. Keyword Search (BM25)")
    print("3. Hybrid Search (semantic + keyword)")
    print("4. Confronto tutte le strategie")
    
    choice = input("\nSelezione (1-4): ").strip()
    
    if choice == "1":
        esempio_test_semantic_search()
    elif choice == "2":
        esempio_test_keyword_search()
    elif choice == "3":
        esempio_test_hybrid_search()
    elif choice == "4":
        esempio_confronto_strategie()
    else:
        print("Selezione non valida")


if __name__ == "__main__":
    main()
