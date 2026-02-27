"""
PIPELINE UNIFICATA DI VALUTAZIONE SISTEMA RAG
==============================================

Pipeline completa per la valutazione quantitativa di sistemi RAG con:
- Integrazione diretta con il sistema RAG esistente (search.py, llm.py)
- Metriche di retrieval (Precision@k, Recall@k, LLM-as-judge)
- Metriche di generation (Faithfulness, Answer Relevancy, Semantic Similarity)
- Supporto per diverse strategie: Semantic, Keyword (BM25), Hybrid

Tesi Magistrale in Ingegneria Informatica e dell'IA
Autore: Andrea Cantelli
Data: Gennaio 2026
"""

import os
import json
import time
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from datetime import datetime
from openai import AzureOpenAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import llm.search as search
import llm.llm as your_llm
from file_embedding.embedding import get_embedding


# ==================== CONFIGURAZIONE ====================

class SearchStrategy:
    SEMANTIC = "semantic"      # Vector similarity
    KEYWORD = "keyword"        # BM25
    HYBRID = "hybrid"          # Combinazione di entrambi
    MULTISTAGE = "multistage"  # Pipeline multi-stage con tool selection adattivo

GOLD_DATASET_PATH = "C:/Users/ACantelli/OneDrive - centrosoftware.com/Documenti/GitHub/progetto-tesi/main/evaluation/gold_dataset.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL_1 = os.getenv("EMBEDDING_MODEL_1")
EMBEDDING_URL_1 = os.getenv("EMBEDDING_URL_1")
EMBEDDING_VERSION_1 = os.getenv("EMBEDDING_VERSION_1")

EMBEDDING_MODEL_2 = os.getenv("EMBEDDING_MODEL_2")
EMBEDDING_URL_2 = os.getenv("EMBEDDING_URL_2")
EMBEDDING_VERSION_2 = os.getenv("EMBEDDING_VERSION_2")

LLM_MODEL = os.getenv("LLM_MODEL")
LLM_URL = os.getenv("LLM_URL")
LLM_VERSION = os.getenv("LLM_VERSION")

# ==================== STRUTTURE DATI ====================

@dataclass
class EvaluationQuery:
    """Query del dataset GOLD con ground truth."""
    query_id: str
    query_text: str
    relevant_chunk_ids: List[str]
    reference_answer: Optional[str] = None

@dataclass
class RetrievalResult:
    """Risultato del retrieval per una query."""
    query_id: str
    retrieved_chunk_ids: List[str]
    retrieved_chunk_texts: List[str]
    retrieval_time: float
    tool_decision: Optional[Dict] = None  # Solo per strategia MULTISTAGE: decisione decide_tools()

@dataclass
class GenerationResult:
    """Risultato della generation per una query."""
    query_id: str
    generated_answer: str
    context_chunks: List[str]
    generation_time: float

@dataclass
class TestResult:
    """Risultato completo di un test con una specifica configurazione."""
    test_id: str
    timestamp: str
    configuration: Dict
    retrieval_metrics: Dict[str, float]
    generation_metrics: Dict[str, float]
    per_query_details: List[Dict]
    total_evaluation_time: float


# ==================== FUNZIONI DI UTILITÀ ====================

# Carica il client AzureOpenAI
def load_openai_client(api_key: Optional[str] = None) -> AzureOpenAI:

    client = AzureOpenAI(
        azure_endpoint=os.getenv("LLM_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("LLM_VERSION")
    )
    return client

# Calcola la similarità coseno tra due vettori
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    return float(np.dot(vec1_np, vec2_np) / (np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np)))

# Carica il dataset GOLD da file JSON
def load_gold_dataset(filepath: str) -> List[EvaluationQuery]:
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = []
    for item in data:
        query = EvaluationQuery(
            query_id=item['query_id'],
            query_text=item['query_text'],
            relevant_chunk_ids=item['relevant_chunk_ids'],
            reference_answer=item.get('reference_answer')
        )
        queries.append(query)
    
    return queries


# ==================== METRICHE DI RETRIEVAL ====================
# TODO: valutarle su span di testo invece che chunk

# Calcola Precision@k - Formula: P@k = (N° documenti rilevanti nei top-k) / k
def calculate_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:

    if k == 0 or len(retrieved_ids) == 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)

    return relevant_in_top_k / k

# Calcola Recall@k - Formula: R@k = (N° documenti rilevanti nei top-k) / (Totale documenti rilevanti)
def calculate_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    
    if len(relevant_ids) == 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    
    return relevant_in_top_k / len(relevant_ids)

# Valuta la rilevanza di un singolo chunk rispetto a una query usando LLM-as-judge
def evaluate_chunk_relevance_with_llm(
    query: str,
    chunk_text: str,
    client: AzureOpenAI,
    model: str = "gpt-4o-mini"
) -> float:

    prompt = f"""Sei un valutatore di rilevanza per sistemi di retrieval.
                Dato:
                - Query utente: "{query}"
                - Chunk di documento: "{chunk_text}"

                Valuta se questo chunk è rilevante per rispondere alla query.

                Rispondi SOLO con un numero tra 0 e 1, dove:
                - 0.0 = completamente irrilevante
                - 0.5 = parzialmente rilevante
                - 1.0 = altamente rilevante

                Risposta (solo numero):"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        
        score_text = response.choices[0].message.content.strip()
        score = float(score_text)
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        print(f"Errore valutazione LLM chunk: {e}")
        return 0.0

# Calcola la rilevanza media dei chunk recuperati usando LLM-as-judge
def calculate_average_chunk_relevance(
    query: str,
    retrieved_chunks: List[str],
    client: AzureOpenAI,
    model: str = "gpt-4o-mini",
    top_k: Optional[int] = None
) -> float:
    
    chunks_to_evaluate = retrieved_chunks[:top_k] if top_k else retrieved_chunks
    
    if not chunks_to_evaluate:
        return 0.0
    
    relevance_scores = []
    for chunk in chunks_to_evaluate:
        score = evaluate_chunk_relevance_with_llm(query, chunk, client, model)
        relevance_scores.append(score)
    
    return float(np.mean(relevance_scores))

# Valuta il retrieval per una singola query usando varie metriche
def evaluate_retrieval_for_query(
    query: EvaluationQuery,
    retrieval_result: RetrievalResult,
    client: AzureOpenAI,
    k_values: List[int] = [1, 3, 5, 10],
    llm_model: str = "gpt-4o-mini"
) -> Dict[str, float]:

    metrics = {}
    
    # Precision e Recall per diversi k
    for k in k_values:
        metrics[f"precision_at_{k}"] = calculate_precision_at_k(
            retrieval_result.retrieved_chunk_ids,
            query.relevant_chunk_ids,
            k
        )
        metrics[f"recall_at_{k}"] = calculate_recall_at_k(
            retrieval_result.retrieved_chunk_ids,
            query.relevant_chunk_ids,
            k
        )
    
    # LLM-as-judge chunk relevance per diversi k
    for k in k_values:
        avg_relevance = calculate_average_chunk_relevance(
            query.query_text,
            retrieval_result.retrieved_chunk_texts,
            client,
            llm_model,
            top_k=k
        )
        metrics[f"avg_chunk_relevance_top{k}"] = avg_relevance
    
    # Retrieval time
    metrics["retrieval_time"] = retrieval_result.retrieval_time
    
    return metrics


# ==================== METRICHE DI GENERATION ====================

# Valuta la faithfulness (fedeltà) della risposta generata rispetto al contesto
def evaluate_faithfulness_with_llm(
    context_chunks: List[str],
    generated_answer: str,
    client: AzureOpenAI,
    model: str = "gpt-4o-mini"
) -> float:

    context_text = "\n\n".join(context_chunks)
    
    prompt = f"""Sei un valutatore di fedeltà per sistemi RAG.

                Dato:
                - CONTESTO (dai documenti):
                {context_text}

                - RISPOSTA GENERATA:
                {generated_answer}

                Valuta se la risposta è FEDELE al contesto, cioè se tutte le informazioni nella risposta sono supportate dal contesto fornito.

                Rispondi SOLO con un numero tra 0 e 1, dove:
                - 0.0 = risposta completamente infedele (inventa informazioni)
                - 0.5 = risposta parzialmente fedele
                - 1.0 = risposta completamente fedele (tutte le info sono nel contesto)

                Risposta (solo numero):"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        
        score_text = response.choices[0].message.content.strip()
        score = float(score_text)
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        print(f"Errore valutazione faithfulness: {e}")
        return 0.0

# Valuta la relevancy (rilevanza) della risposta rispetto alla query
def evaluate_answer_relevancy_with_llm(
    query: str,
    generated_answer: str,
    client: AzureOpenAI,
    model: str = "gpt-4o-mini"
) -> float:

    prompt = f"""Sei un valutatore di rilevanza per risposte generate.

                Dato:
                - QUERY UTENTE: "{query}"
                - RISPOSTA: "{generated_answer}"

                Valuta se la risposta è RILEVANTE per la query, cioè se risponde effettivamente alla domanda posta.

                Rispondi SOLO con un numero tra 0 e 1, dove:
                - 0.0 = risposta completamente irrilevante
                - 0.5 = risposta parzialmente rilevante
                - 1.0 = risposta perfettamente rilevante

                Risposta (solo numero):"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        
        score_text = response.choices[0].message.content.strip()
        score = float(score_text)
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        print(f"Errore valutazione answer relevancy: {e}")
        return 0.0

# Calcola la similarità semantica tra risposta di riferimento e risposta generata come metrica
def calculate_semantic_similarity_embeddings(
    reference_answer: str,
    generated_answer: str,
) -> float:

    try:
        # Genera embeddings per entrambe le risposte
        embedding_ref = get_embedding(reference_answer)
        embedding_gen = get_embedding(generated_answer)

        return cosine_similarity(embedding_ref, embedding_gen)
        
    except Exception as e:
        print(f"Errore calcolo semantic similarity: {e}")
        print(f"  - Reference answer type: {type(reference_answer)}, value: {reference_answer[:100] if reference_answer else 'None'}...")
        print(f"  - Generated answer type: {type(generated_answer)}, value: {generated_answer[:100] if generated_answer else 'None'}...")
        return 0.0

# Valuta la generation per una singola query usando varie metriche
def evaluate_generation_for_query(
    query: EvaluationQuery,
    generation_result: GenerationResult,
    client: AzureOpenAI,
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-large"
) -> Dict[str, float]:
   
    metrics = {}
    
    # Faithfulness
    metrics["faithfulness"] = evaluate_faithfulness_with_llm(
        generation_result.context_chunks,
        generation_result.generated_answer,
        client,
        llm_model
    )
    
    # Answer Relevancy
    metrics["answer_relevancy"] = evaluate_answer_relevancy_with_llm(
        query.query_text,
        generation_result.generated_answer,
        client,
        llm_model
    )
    
    # Semantic Similarity
    if query.reference_answer:
        metrics["semantic_similarity"] = calculate_semantic_similarity_embeddings(
            query.reference_answer,
            generation_result.generated_answer
        )
    else:
        metrics["semantic_similarity"] = None
    
    # Generation time
    metrics["generation_time"] = generation_result.generation_time
    
    return metrics


# ==================== AGGREGAZIONE METRICHE ====================
def aggregate_retrieval_metrics(per_query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_query_metrics:
        return {}
    
    # Estrai tutte le chiavi di metrica
    metric_keys = set()
    for metrics in per_query_metrics:
        metric_keys.update(metrics.keys())
    
    # Calcola media per ogni metrica
    aggregated = {}
    for key in metric_keys:
        values = [m[key] for m in per_query_metrics if key in m and m[key] is not None]
        if values:
            aggregated[f"avg_{key}"] = float(np.mean(values))
    
    return aggregated

def aggregate_generation_metrics(per_query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_query_metrics:
        return {}
    
    # Estrai tutte le chiavi di metrica
    metric_keys = set()
    for metrics in per_query_metrics:
        metric_keys.update(metrics.keys())
    
    # Calcola media per ogni metrica
    aggregated = {}
    for key in metric_keys:
        values = [m[key] for m in per_query_metrics if key in m and m[key] is not None]
        if values:
            aggregated[f"avg_{key}"] = float(np.mean(values))
    
    return aggregated


# ==================== INTEGRAZIONE CON SISTEMA RAG ESISTENTE ====================

# Esegue retrieval usando la funzione semantic_search()
def run_retrieval_with_semantic_search(
    queries: List[EvaluationQuery],
    top_k: int = 10
) -> List[RetrievalResult]:

    retrieval_results = []
    
    print(f"\n{'='*70}")
    print(f"RETRIEVAL SEMANTICO: {len(queries)} query")
    print(f"{'='*70}\n")
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query.query_text[:60]}...")
        
        start_time = time.time()
        
        try:
            docs = search.semantic_search(
                prompt=query.query_text,
                top_n=top_k
            )
            
            # Estrai ID e testi
            retrieved_chunk_ids = [
                f"{doc['numero']}_{doc['progressivo']}" 
                for doc in docs
            ]
            retrieved_chunk_texts = [doc['content'] for doc in docs]
            
            print(f"  ✓ {len(retrieved_chunk_ids)} chunk recuperati tramite retrieval semantico")
            
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

# Esegue retrieval usando la funzione keyword_search() (BM25)
def run_retrieval_with_keyword_search(
    queries: List[EvaluationQuery],
    top_k: int = 10
) -> List[RetrievalResult]:
    
    retrieval_results = []
    
    print(f"\n{'='*70}")
    print(f"RETRIEVAL BM25: {len(queries)} query")
    print(f"{'='*70}\n")
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query.query_text[:60]}...")
        
        start_time = time.time()
        
        try:
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
            
            print(f"  ✓ {len(retrieved_chunk_ids)} chunk recuperati tramite retrieval keyword")
            
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

# Esegue retrieval ibrido: combina semantic e keyword search (semantic_weight per cambiare % impatto di ciascuna)
def run_retrieval_hybrid(
    queries: List[EvaluationQuery],
    top_k: int = 10,
    semantic_weight: float = 0.7
) -> List[RetrievalResult]:
    
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
            semantic_docs = search.semantic_search(query.query_text, top_n=top_k*2)
            keyword_docs = search.keyword_search(query.query_text, top_n=top_k*2)
            
            # Fonde risultati con weighted score
            combined = {}
            
            for doc in semantic_docs:
                chunk_id = f"{doc['numero']}_{doc['progressivo']}"
                combined[chunk_id] = {
                    'doc': doc,
                    'score': doc['similarity'] * semantic_weight
                }
            
            for doc in keyword_docs:
                chunk_id = f"{doc['numero']}_{doc['progressivo']}"
                # Normalizza BM25 score
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
            
            print(f"  ✓ {len(retrieved_chunk_ids)} chunk recuperati tramite retrieval ibrido")
            
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

# Esegue retrieval con pipeline multi-stage (tool selection adattivo)
def run_retrieval_with_multistage(
    queries: List[EvaluationQuery],
    top_k: int = 10
) -> List[RetrievalResult]:
    """
    Retrieval adattivo: per ogni query chiama decide_tools() per scegliere
    quale retriever usare (semantic, keyword o entrambi), tracciando le decisioni
    per calcolare la distribuzione dell'utilizzo degli strumenti.
    """

    retrieval_results = []

    print(f"\n{'='*70}")
    print(f"RETRIEVAL MULTI-STAGE (adattivo): {len(queries)} query")
    print(f"{'='*70}\n")

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query.query_text[:60]}...")

        start_time = time.time()

        try:
            # Step 1: tool selection
            tool_decision = your_llm.decide_tools(query.query_text)
            use_semantic = tool_decision.get("use_semantic", False)
            use_keyword = tool_decision.get("use_keyword", False)

            # Step 2: retrieval in base alla decisione
            all_docs = []
            if use_semantic:
                semantic_docs = search.semantic_search(query.query_text, top_n=top_k)
                for doc in semantic_docs:
                    doc["retrieval_sources"] = ["semantic"]
                all_docs += semantic_docs

            if use_keyword:
                keyword_docs = search.keyword_search(query.query_text, top_n=top_k)
                for doc in keyword_docs:
                    doc["retrieval_sources"] = ["keyword"]
                all_docs += keyword_docs

            # Deduplicazione con merge sorgenti
            seen = {}
            
            total_before_dedup = len(all_docs) # ADDED

            for doc in all_docs:
                key = (doc["numero"], doc["progressivo"])
                if key not in seen:
                    seen[key] = doc
                else:
                    existing = seen[key].get("retrieval_sources", [])
                    incoming = doc.get("retrieval_sources", [])
                    seen[key]["retrieval_sources"] = list(set(existing + incoming))
                    if doc.get("similarity", 0) > seen[key].get("similarity", 0):
                        seen[key]["similarity"] = doc["similarity"]
            all_docs = list(seen.values())
            
            total_after_dedup = len(all_docs) # ADDED
            co_retrieved_count = total_before_dedup - total_after_dedup
            if total_before_dedup > 0:
                co_retrieved_pct = (co_retrieved_count / total_before_dedup) * 100
                print(f"  → Co-retrieval: {co_retrieved_count}/{total_before_dedup} chunk ({co_retrieved_pct:.1f}%)")

            retrieved_chunk_ids = [
                f"{doc['numero']}_{doc['progressivo']}"
                for doc in all_docs
            ]
            retrieved_chunk_texts = [doc['content'] for doc in all_docs]

            # Log decisione
            if use_semantic and use_keyword:
                mode = "HYBRID (semantic + keyword)"
            elif use_semantic:
                mode = "SEMANTIC only"
            elif use_keyword:
                mode = "KEYWORD only"
            else:
                mode = "NESSUNA RICERCA"

            print(f"  → Tool selection: {mode}")
            print(f"  → Reason: {tool_decision.get('reason', '')[:80]}")
            print(f"  ✓ {len(retrieved_chunk_ids)} chunk recuperati")

        except Exception as e:
            print(f"  ✗ Errore: {e}")
            retrieved_chunk_ids = []
            retrieved_chunk_texts = []
            tool_decision = {"use_semantic": False, "use_keyword": False, "reason": f"Errore: {e}"}

        retrieval_time = time.time() - start_time

        result = RetrievalResult(
            query_id=query.query_id,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_chunk_texts=retrieved_chunk_texts,
            retrieval_time=retrieval_time,
            tool_decision=tool_decision
        )
        retrieval_results.append(result)

    print(f"\n{'='*70}\n")
    return retrieval_results


# Aggrega le statistiche di tool selection per la strategia MULTISTAGE
def aggregate_tool_selection_stats(retrieval_results: List[RetrievalResult]) -> Dict:
    """
    Calcola la distribuzione delle decisioni di tool selection su tutte le query.
    Utile per giustificare empiricamente il valore dello step di selezione adattiva
    rispetto all'approccio hybrid fisso.
    """

    decisions = [r.tool_decision for r in retrieval_results if r.tool_decision is not None]

    if not decisions:
        return {}

    total = len(decisions)
    semantic_only  = sum(1 for d in decisions if d.get("use_semantic") and not d.get("use_keyword"))
    keyword_only   = sum(1 for d in decisions if d.get("use_keyword") and not d.get("use_semantic"))
    both           = sum(1 for d in decisions if d.get("use_semantic") and d.get("use_keyword"))
    none_selected  = sum(1 for d in decisions if not d.get("use_semantic") and not d.get("use_keyword"))

    stats = {
        "total_queries": total,
        "semantic_only_count": semantic_only,
        "keyword_only_count": keyword_only,
        "both_count": both,
        "none_count": none_selected,
        "pct_semantic_only": round(semantic_only / total * 100, 1),
        "pct_keyword_only": round(keyword_only / total * 100, 1),
        "pct_both": round(both / total * 100, 1),
        "pct_none": round(none_selected / total * 100, 1),
        # Quante volte è stato evitato di usare entrambi (risparmio rispetto a hybrid fisso)
        "pct_avoided_hybrid": round((semantic_only + keyword_only + none_selected) / total * 100, 1),
        "per_query_decisions": [
            {
                "query_id": r.query_id,
                "use_semantic": r.tool_decision.get("use_semantic"),
                "use_keyword": r.tool_decision.get("use_keyword"),
                "mode": (
                    "both" if r.tool_decision.get("use_semantic") and r.tool_decision.get("use_keyword")
                    else "semantic_only" if r.tool_decision.get("use_semantic")
                    else "keyword_only" if r.tool_decision.get("use_keyword")
                    else "none"
                ),
                "reason": r.tool_decision.get("reason", "")
            }
            for r in retrieval_results if r.tool_decision is not None
        ]
    }

    return stats


# Esegue generation della risposta tramite LLM
def run_generation_with_llm(
    queries: List[EvaluationQuery],
    retrieval_results: List[RetrievalResult],
    llm_model: str = "gpt-4o-mini"
) -> List[GenerationResult]:

    generation_results = []
    
    print(f"\n{'='*70}")
    print(f"GENERATION: {len(queries)} query")
    print(f"{'='*70}\n")
    
    for i, (query, retrieval_result) in enumerate(zip(queries, retrieval_results), 1):
        print(f"[{i}/{len(queries)}] {query.query_text[:60]}...")
        
        start_time = time.time()
        
        chunk_ids = retrieval_result.retrieved_chunk_ids
        chunk_texts = retrieval_result.retrieved_chunk_texts
        
        try:
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
            
            answer = your_llm.generate_final_answer(
                user_prompt=query.query_text,
                selected_docs=fake_docs,
                chat_history=[]
            )

            # Gestisci generator
            import types
            if isinstance(answer, types.GeneratorType):
                answer = ''.join(answer)
                print(f"  ✓ Risposta generata (da stream): {len(answer)} char")
            elif isinstance(answer, str):
                print(f"  ✓ Risposta generata: {len(answer)} char")
            else:
                answer = str(answer)
                print(f"  ⚠ Tipo inaspettato: {len(answer)} char")

            if not answer or not answer.strip():
                answer = "[RISPOSTA VUOTA]"
            
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


# ==================== VALUTAZIONE COMPLETA ====================

# Esegue la valutazione completa di retrieval e generation
def run_full_evaluation(
    gold_queries: List[EvaluationQuery],
    retrieval_results: List[RetrievalResult],
    generation_results: List[GenerationResult],
    configuration: Dict,
    client: AzureOpenAI,
    k_values: List[int] = [5, 10],
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-large"
) -> TestResult:

    start_time = time.time()
    
    print("\n" + "="*70)
    print("VALUTAZIONE METRICHE")
    print("="*70)
    
    per_query_retrieval_metrics = []
    per_query_generation_metrics = []
    per_query_details = []
    
    total_queries = len(gold_queries)
    
    for i, (query, retrieval_result, generation_result) in enumerate(
        zip(gold_queries, retrieval_results, generation_results), 1
    ):
        print(f"\n[{i}/{total_queries}] Valutazione query: {query.query_id}")
        
        # Valuta retrieval
        print("  - Metriche retrieval...")
        retrieval_metrics = evaluate_retrieval_for_query(
            query, retrieval_result, client, k_values, llm_model
        )
        per_query_retrieval_metrics.append(retrieval_metrics)
        
        # Valuta generation
        print("  - Metriche generation...")
        generation_metrics = evaluate_generation_for_query(
            query, generation_result, client, llm_model, embedding_model
        )
        per_query_generation_metrics.append(generation_metrics)
        
        # Salva dettagli per query
        per_query_details.append({
            "query_id": query.query_id,
            "query_text": query.query_text,
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
            "generated_answer": generation_result.generated_answer,
            "retrieved_chunk_count": len(retrieval_result.retrieved_chunk_ids)
        })
        
        print(f"  ✓ Query {query.query_id} completata")
    
    # Aggrega metriche
    print("\n" + "="*70)
    print("AGGREGAZIONE METRICHE")
    print("="*70)
    
    aggregated_retrieval = aggregate_retrieval_metrics(per_query_retrieval_metrics)
    aggregated_generation = aggregate_generation_metrics(per_query_generation_metrics)
    
    total_time = time.time() - start_time
    
    # Creazione risultato finale
    test_result = TestResult(
        test_id=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        timestamp=datetime.now().isoformat(),
        configuration=configuration,
        retrieval_metrics=aggregated_retrieval,
        generation_metrics=aggregated_generation,
        per_query_details=per_query_details,
        total_evaluation_time=total_time
    )
    
    print(f"\n✓ Valutazione completata in {total_time:.2f}s")
    print("="*70)
    
    return test_result


# ==================== GESTIONE RISULTATI ====================

# Salva il risultato del test in un file JSON
def save_test_result(test_result: TestResult, output_dir: str = "evaluation_results"):

    Path(output_dir).mkdir(exist_ok=True)
    
    filename = f"{test_result.test_id}.json"
    filepath = Path(output_dir) / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(asdict(test_result), f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Risultati salvati: {filepath}")

def print_test_result(test_result: TestResult):

    print("\n" + "="*70)
    print(f"TEST REPORT: {test_result.test_id}")
    print("="*70)
    
    print(f"\n📅 Timestamp: {test_result.timestamp}")
    print(f"⏱️  Tempo totale: {test_result.total_evaluation_time:.2f}s")
    
    print("\n📋 CONFIGURAZIONE:")
    for key, value in test_result.configuration.items():
        print(f"  • {key}: {value}")
    
    print("\n📊 METRICHE RETRIEVAL:")
    for metric, value in test_result.retrieval_metrics.items():
        print(f"  • {metric}: {value:.4f}")
    
    print("\n📝 METRICHE GENERATION:")
    for metric, value in test_result.generation_metrics.items():
        print(f"  • {metric}: {value:.4f}")
    
    print("\n" + "="*70)

# Genera una tabella comparativa di più risultati di test
def compare_test_results(test_results: List[TestResult]):

    if not test_results:
        print("Nessun risultato da confrontare.")
        return
    
    print("\n" + "="*120)
    print("CONFRONTO RISULTATI TEST")
    print("="*120)
    
    # Header
    config_name_len = 40
    print(f"\n{'Test / Config':<{config_name_len}} {'P@5':>8} {'R@5':>8} {'LLM-Rel':>8} {'Faith':>8} {'Relev':>8} {'SemSim':>8}")
    print("-"*120)
    
    # Riga per ogni test
    for result in test_results:
        # Nome configurazione (truncated)
        config_name = result.configuration.get('name', result.test_id)[:config_name_len-2]
        
        # Estrazione metriche
        p5 = result.retrieval_metrics.get('avg_precision_at_5', 0.0)
        r5 = result.retrieval_metrics.get('avg_recall_at_5', 0.0)
        llm_rel = result.retrieval_metrics.get('avg_avg_chunk_relevance_top5', 0.0)
        faith = result.generation_metrics.get('avg_faithfulness', 0.0)
        relev = result.generation_metrics.get('avg_answer_relevancy', 0.0)
        simsem = result.generation_metrics.get('avg_semantic_similarity', 0.0)
        
        print(f"{config_name:<{config_name_len}} {p5:>8.4f} {r5:>8.4f} {llm_rel:>8.4f} {faith:>8.4f} {relev:>8.4f} {simsem:>8.4f}")
    
    print("="*120)
    print("\nLegenda:")
    print("  P@5      = Precision at 5")
    print("  R@5      = Recall at 5")
    print("  LLM-Rel  = LLM-as-judge Chunk Relevance (avg top 5)")
    print("  Faith    = Faithfulness (LLM-as-judge)")
    print("  Relev    = Answer Relevancy (LLM-as-judge)")
    print("  SemSim   = Semantic Similarity (embeddings)")
    print()


# ==================== TEST PRINCIPALE ====================

# Test completo di valutazione
def run_test_with_your_pipeline(
    gold_dataset_path: str,
    configuration: dict,
    search_strategy: str = SearchStrategy.SEMANTIC,
    results_dir: str = "evaluation_results"
):

    print("\n" + "="*70)
    print(f"TEST: {configuration.get('name', 'Unnamed')}")
    print(f"Search Strategy: {search_strategy}")
    print("="*70)
    
    # 1. Carica dataset
    print("\n[1/4] Caricamento dataset GOLD...")
    gold_queries = load_gold_dataset(gold_dataset_path)
    
    # 2. Retrieval
    print(f"\n[2/4] Retrieval con strategia: {search_strategy}...")
    top_k = configuration.get('top_k', 10)
    
    if search_strategy == SearchStrategy.SEMANTIC:
        retrieval_results = run_retrieval_with_semantic_search(gold_queries, top_k)
    elif search_strategy == SearchStrategy.KEYWORD:
        retrieval_results = run_retrieval_with_keyword_search(gold_queries, top_k)
    elif search_strategy == SearchStrategy.HYBRID:
        retrieval_results = run_retrieval_hybrid(gold_queries, top_k, semantic_weight=configuration.get('semantic_weight', 0.7)
        )
    elif search_strategy == SearchStrategy.MULTISTAGE:
        retrieval_results = run_retrieval_with_multistage(gold_queries, top_k)
    else:
        raise ValueError(f"Search strategy non valida: {search_strategy}")
    
    # 3. Generation
    print("\n[3/4] Generation...")
    generation_results = run_generation_with_llm(
        gold_queries,
        retrieval_results,
        llm_model=configuration.get('llm_model', 'gpt-4-1')
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
        k_values=[3, 5, 10],
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-large"
    )
    
    print_test_result(test_result)

    # Per strategia multistage: stampa e salva statistiche tool selection
    if search_strategy == SearchStrategy.MULTISTAGE:
        tool_stats = aggregate_tool_selection_stats(retrieval_results)
        print_tool_selection_stats(tool_stats)
        # Aggiunge le stats alla configurazione del test result per averle nel JSON
        test_result.configuration["tool_selection_stats"] = tool_stats

    save_test_result(test_result, results_dir)
    
    return test_result


# ==================== ESEMPI DI TEST ====================

def print_tool_selection_stats(stats: Dict):
    """Stampa le statistiche di tool selection per la strategia MULTISTAGE."""

    if not stats:
        return

    print("\n" + "="*70)
    print("📊 STATISTICHE TOOL SELECTION (Multi-stage)")
    print("="*70)
    print(f"\n  Totale query analizzate : {stats['total_queries']}")
    print(f"\n  Semantic only           : {stats['semantic_only_count']:>3} query  ({stats['pct_semantic_only']:>5.1f}%)")
    print(f"  Keyword only            : {stats['keyword_only_count']:>3} query  ({stats['pct_keyword_only']:>5.1f}%)")
    print(f"  Entrambi (≈ hybrid)     : {stats['both_count']:>3} query  ({stats['pct_both']:>5.1f}%)")
    print(f"  Nessuna ricerca         : {stats['none_count']:>3} query  ({stats['pct_none']:>5.1f}%)")
    print(f"\n  → Risparmio vs. hybrid fisso: {stats['pct_avoided_hybrid']:.1f}% delle query")
    print(f"    ha evitato di eseguire entrambi i retriever\n")

    print("  Dettaglio per query:")
    print(f"  {'Query ID':<12} {'Mode':<16} {'Reason'}")
    print("  " + "-"*66)
    for d in stats.get("per_query_decisions", []):
        reason_short = d['reason'][:45] + "..." if len(d['reason']) > 45 else d['reason']
        print(f"  {d['query_id']:<12} {d['mode']:<16} {reason_short}")
    print("="*70)


# Test con SOLO semantic search (vector similarity)
def esempio_test_semantic_search():

    configuration = {
        "name": "Test Semantic Search",
        "search_strategy": "semantic",
        "embedding_model": EMBEDDING_MODEL_1,
        "llm_model": "gpt-4.1",
        "top_k": 7,
        "note": "Vector similarity con embeddings Azure"
    }
    
    return run_test_with_your_pipeline(
        gold_dataset_path=GOLD_DATASET_PATH,
        configuration=configuration,
        search_strategy=SearchStrategy.SEMANTIC
    )

# Test con SOLO keyword search (BM25)
def esempio_test_keyword_search():

    configuration = {
        "name": "Test Keyword Search BM25",
        "search_strategy": "keyword",
        "llm_model": "gpt-4.1",
        "top_k": 7,
        "note": "BM25 keyword search con stemming italiano"
    }
    
    return run_test_with_your_pipeline(
        gold_dataset_path=GOLD_DATASET_PATH,
        configuration=configuration,
        search_strategy=SearchStrategy.KEYWORD
    )

# Test con ricerca IBRIDA (semantic + keyword)
def esempio_test_hybrid_search():

    configuration = {
        "name": "Test Hybrid Search",
        "search_strategy": "hybrid",
        "semantic_weight": 0.5,
        "keyword_weight": 0.5,
        "embedding_model": EMBEDDING_MODEL_1,
        "llm_model": "gpt-4.1",
        "top_k": 10,
        "note": "Combinazione semantic (70%) + keyword (30%)"
    }
    
    return run_test_with_your_pipeline(
        gold_dataset_path=GOLD_DATASET_PATH,
        configuration=configuration,
        search_strategy=SearchStrategy.HYBRID
    )

# Test con pipeline MULTI-STAGE (tool selection adattivo)
def esempio_test_multistage():

    configuration = {
        "name": "Test Multi-stage (adattivo)",
        "search_strategy": "multistage",
        "embedding_model": EMBEDDING_MODEL_1,
        "llm_model": "gpt-4.1",
        "top_k": 10,
        "note": "Tool selection adattivo via LLM + deduplicazione con merge sorgenti"
    }

    return run_test_with_your_pipeline(
        gold_dataset_path=GOLD_DATASET_PATH,
        configuration=configuration,
        search_strategy=SearchStrategy.MULTISTAGE
    )


# Confronto diretto delle 3 strategie di search
def esempio_confronto_strategie():

    print("\n" + "="*70)
    print("CONFRONTO STRATEGIE DI SEARCH")
    print("="*70)
    
    results = []
    
    # Test 1: Semantic
    print("\n[TEST 1/4] Semantic Search...")
    results.append(esempio_test_semantic_search())
    
    # Test 2: Keyword
    print("\n[TEST 2/4] Keyword Search (BM25)...")
    results.append(esempio_test_keyword_search())
    
    # Test 3: Hybrid
    print("\n[TEST 3/4] Hybrid Search...")
    results.append(esempio_test_hybrid_search())

    # Test 4: Multi-stage
    print("\n[TEST 4/4] Multi-stage (adattivo)...")
    results.append(esempio_test_multistage())
    
    compare_test_results(results)


# ==================== MAIN ====================

def main():
    """
    Menu interattivo per scegliere il test da eseguire.
    """
    print("="*70)
    print("PIPELINE UNIFICATA DI VALUTAZIONE SISTEMA RAG")
    print("="*70)
    
    print("\nScegli il tipo di test:")
    print("1. Semantic Search (vector similarity)")
    print("2. Keyword Search (BM25)")
    print("3. Hybrid Search (semantic + keyword)")
    print("4. Multi-stage (tool selection adattivo)")
    print("5. Confronto tutte le strategie")
    
    choice = input("\nSelezione (1-5): ").strip()
    
    if choice == "1":
        esempio_test_semantic_search()
    elif choice == "2":
        esempio_test_keyword_search()
    elif choice == "3":
        esempio_test_hybrid_search()
    elif choice == "4":
        esempio_test_multistage()
    elif choice == "5":
        esempio_confronto_strategie()
    else:
        print("Selezione non valida")


if __name__ == "__main__":
    main()