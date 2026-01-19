"""
Pipeline di Valutazione per Sistema RAG
Tesi Magistrale in Ingegneria Informatica e dell'IA

Implementazione funzionale di metriche di valutazione per sistemi RAG
con utilizzo di LLM-as-judge tramite API AzureOpenAI.

Autore: [Nome Studente]
Data: Gennaio 2026
"""

import os
import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from datetime import datetime

# Import per AzureOpenAI e embeddings
try:
    from openai import AzureOpenAI
except ImportError:
    print("ATTENZIONE: Installare openai")

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
    retrieved_chunk_texts: List[str]  # Testi dei chunk per LLM-as-judge
    retrieval_time: float


@dataclass
class GenerationResult:
    """Risultato della generation per una query."""
    query_id: str
    generated_answer: str
    context_chunks: List[str]  # Chunk usati come contesto
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


# ==================== CONFIGURAZIONE ====================

def load_openai_client(api_key: Optional[str] = None) -> AzureOpenAI:
    """
    Carica il client AzureOpenAI.
    
    Args:
        api_key: Chiave API (se None, usa variabile ambiente OPENAI_API_KEY)
        
    Returns:
        Client AzureOpenAI configurato
    """
    client = AzureOpenAI(
        azure_endpoint=os.getenv("LLM_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("LLM_VERSION")
    )

    return client


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcola la similarità coseno tra due vettori.
    
    Args:
        vec1: Primo vettore
        vec2: Secondo vettore
        
    Returns:
        Similarità coseno (0-1)
    """
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    return float(np.dot(vec1_np, vec2_np) / (np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np)))


# ==================== RETRIEVAL METRICS ====================

def calculate_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Calcola Precision@k.
    
    Formula: P@k = (N° documenti rilevanti nei top-k) / k
    
    Args:
        retrieved_ids: Lista ordinata degli ID recuperati
        relevant_ids: Lista degli ID rilevanti (ground truth)
        k: Numero di top risultati da considerare
        
    Returns:
        Precision@k (0-1)
    """
    if k == 0 or len(retrieved_ids) == 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    
    return relevant_in_top_k / k


def calculate_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Calcola Recall@k.
    
    Formula: R@k = (N° documenti rilevanti nei top-k) / (Totale documenti rilevanti)
    
    Args:
        retrieved_ids: Lista ordinata degli ID recuperati
        relevant_ids: Lista degli ID rilevanti (ground truth)
        k: Numero di top risultati da considerare
        
    Returns:
        Recall@k (0-1)
    """
    if len(relevant_ids) == 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    
    return relevant_in_top_k / len(relevant_ids)


def evaluate_chunk_relevance_with_llm(
    query: str,
    chunk_text: str,
    client: AzureOpenAI,
    model: str = "gpt-4o-mini"
) -> float:
    """
    Valuta la rilevanza di un chunk rispetto a una query usando LLM-as-judge.
    
    Args:
        query: Query dell'utente
        chunk_text: Testo del chunk da valutare
        client: Client AzureOpenAI
        model: Modello LLM da usare
        
    Returns:
        Score di rilevanza 0-1
    """
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
        return max(0.0, min(1.0, score))  # Clamp tra 0 e 1
        
    except Exception as e:
        print(f"Errore valutazione LLM chunk: {e}")
        return 0.0


def calculate_average_chunk_relevance(
    query: str,
    retrieved_chunks: List[str],
    client: AzureOpenAI,
    model: str = "gpt-4o-mini",
    top_k: Optional[int] = None
) -> float:
    """
    Calcola la rilevanza media dei chunk recuperati usando LLM-as-judge.
    
    Args:
        query: Query dell'utente
        retrieved_chunks: Lista dei testi dei chunk recuperati
        client: Client AzureOpenAI
        model: Modello LLM da usare
        top_k: Se specificato, considera solo i primi k chunk
        
    Returns:
        Score medio di rilevanza (0-1)
    """
    chunks_to_evaluate = retrieved_chunks[:top_k] if top_k else retrieved_chunks
    
    if not chunks_to_evaluate:
        return 0.0
    
    relevance_scores = []
    for chunk in chunks_to_evaluate:
        score = evaluate_chunk_relevance_with_llm(query, chunk, client, model)
        relevance_scores.append(score)
    
    return float(np.mean(relevance_scores))


def evaluate_retrieval_for_query(
    query: EvaluationQuery,
    retrieval_result: RetrievalResult,
    client: AzureOpenAI,
    k_values: List[int] = [1, 3, 5, 10],
    llm_model: str = "gpt-4o-mini"
) -> Dict[str, float]:
    """
    Valuta il retrieval per una singola query.
    
    Args:
        query: Query con ground truth
        retrieval_result: Risultato del retrieval
        client: Client AzureOpenAI per LLM-as-judge
        k_values: Valori di k per precision/recall
        llm_model: Modello LLM per valutazione
        
    Returns:
        Dizionario con metriche di retrieval
    """
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
    
    # LLM-as-judge chunk relevance (sui top 5)
    metrics["avg_chunk_relevance_top5"] = calculate_average_chunk_relevance(
        query.query_text,
        retrieval_result.retrieved_chunk_texts,
        client,
        llm_model,
        top_k=5
    )
    
    metrics["retrieval_time"] = retrieval_result.retrieval_time
    
    return metrics


def aggregate_retrieval_metrics(per_query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggrega le metriche di retrieval da tutte le query.
    
    Args:
        per_query_metrics: Lista di dizionari con metriche per query
        
    Returns:
        Dizionario con metriche aggregate (medie)
    """
    if not per_query_metrics:
        return {}
    
    # Raccogli tutti i nomi delle metriche
    metric_names = per_query_metrics[0].keys()
    
    aggregated = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in per_query_metrics]
        aggregated[f"avg_{metric_name}"] = float(np.mean(values))
    
    return aggregated


# ==================== GENERATION METRICS ====================

def evaluate_faithfulness_with_llm(
    generated_answer: str,
    context_chunks: List[str],
    client: AzureOpenAI,
    model: str = "gpt-4o-mini"
) -> float:
    """
    Valuta la fedeltà della risposta al contesto usando LLM-as-judge.
    
    L'LLM verifica se tutte le affermazioni nella risposta sono supportate dal contesto.
    
    Args:
        generated_answer: Risposta generata dal sistema
        context_chunks: Chunk di contesto utilizzati
        client: Client AzureOpenAI
        model: Modello LLM da usare
        
    Returns:
        Score di faithfulness (0-1)
    """
    context_text = "\n\n".join([f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])
    
    prompt = f"""Sei un valutatore di fedeltà per sistemi RAG.

Contesto fornito:
{context_text}

Risposta generata:
"{generated_answer}"

Valuta se TUTTE le affermazioni nella risposta sono supportate dal contesto fornito.

Rispondi SOLO con un numero tra 0 e 1, dove:
- 0.0 = risposta contiene affermazioni non supportate o contraddittorie
- 0.5 = risposta parzialmente supportata dal contesto
- 1.0 = tutte le affermazioni sono completamente supportate dal contesto

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


def evaluate_answer_relevancy_with_llm(
    query: str,
    generated_answer: str,
    client: AzureOpenAI,
    model: str = "gpt-4o-mini"
) -> float:
    """
    Valuta la rilevanza della risposta rispetto alla query usando LLM-as-judge.
    
    Args:
        query: Query originale dell'utente
        generated_answer: Risposta generata dal sistema
        client: Client AzureOpenAI
        model: Modello LLM da usare
        
    Returns:
        Score di relevancy (0-1)
    """
    prompt = f"""Sei un valutatore di rilevanza per sistemi di Question Answering.

Query utente: "{query}"

Risposta generata: "{generated_answer}"

Valuta quanto la risposta è pertinente e risponde direttamente alla query.

Rispondi SOLO con un numero tra 0 e 1, dove:
- 0.0 = risposta completamente fuori tema
- 0.5 = risposta parzialmente pertinente ma incompleta
- 1.0 = risposta perfettamente pertinente e completa

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
        print(f"Errore valutazione relevancy: {e}")
        return 0.0


def evaluate_semantic_similarity_with_embeddings(
    generated_answer: str,
    reference_answer: str,
    client: AzureOpenAI,
    embedding_model: str = "text-embedding-3-large"
) -> float:
    """
    Valuta la similarità semantica con la risposta di riferimento usando embeddings.
    
    Args:
        generated_answer: Risposta generata dal sistema
        reference_answer: Risposta di riferimento (ground truth)
        client: Client AzureOpenAI
        embedding_model: Modello di embedding da usare
        
    Returns:
        Score di similarità (0-1)
    """
    if not reference_answer:
        return 0.0
    
    try:
        from file_embedding.embedding import get_embedding
        emb_generated = get_embedding(generated_answer)
        emb_reference = get_embedding(reference_answer)
        
        similarity = cosine_similarity(emb_generated, emb_reference)
        return similarity
        
    except Exception as e:
        print(f"Errore calcolo semantic similarity: {e}")
        return 0.0


def evaluate_generation_for_query(
    query: EvaluationQuery,
    generation_result: GenerationResult,
    client: AzureOpenAI,
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-large"
) -> Dict[str, float]:
    """
    Valuta la generation per una singola query.
    
    Args:
        query: Query con ground truth
        generation_result: Risultato della generation
        client: Client AzureOpenAI
        llm_model: Modello LLM per LLM-as-judge
        embedding_model: Modello per embeddings
        
    Returns:
        Dizionario con metriche di generation
    """
    metrics = {}
    
    # Faithfulness (LLM-as-judge)
    metrics["faithfulness"] = evaluate_faithfulness_with_llm(
        generation_result.generated_answer,
        generation_result.context_chunks,
        client,
        llm_model
    )
    
    # Answer Relevancy (LLM-as-judge)
    metrics["answer_relevancy"] = evaluate_answer_relevancy_with_llm(
        query.query_text,
        generation_result.generated_answer,
        client,
        llm_model
    )
    
    # Semantic Similarity (embeddings)
    if query.reference_answer:
        metrics["semantic_similarity"] = evaluate_semantic_similarity_with_embeddings(
            generation_result.generated_answer,
            query.reference_answer,
            client,
            embedding_model
        )
    else:
        metrics["semantic_similarity"] = 0.0
    
    metrics["generation_time"] = generation_result.generation_time
    
    return metrics


def aggregate_generation_metrics(per_query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggrega le metriche di generation da tutte le query.
    
    Args:
        per_query_metrics: Lista di dizionari con metriche per query
        
    Returns:
        Dizionario con metriche aggregate (medie)
    """
    if not per_query_metrics:
        return {}
    
    metric_names = per_query_metrics[0].keys()
    
    aggregated = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in per_query_metrics]
        aggregated[f"avg_{metric_name}"] = float(np.mean(values))
    
    return aggregated


# ==================== PIPELINE PRINCIPALE ====================

def load_gold_dataset(filepath: str) -> List[EvaluationQuery]:
    """
    Carica il dataset GOLD da file JSON.
    
    Args:
        filepath: Percorso al file JSON
        
    Returns:
        Lista di EvaluationQuery
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = [
        EvaluationQuery(
            query_id=item['query_id'],
            query_text=item['query_text'],
            relevant_chunk_ids=item['relevant_chunk_ids'],
            reference_answer=item.get('reference_answer')
        )
        for item in data
    ]
    
    print(f"✓ Dataset GOLD caricato: {len(queries)} query")
    return queries


def run_full_evaluation(
    gold_queries: List[EvaluationQuery],
    retrieval_results: List[RetrievalResult],
    generation_results: List[GenerationResult],
    configuration: Dict,
    client: AzureOpenAI,
    k_values: List[int] = [1, 3, 5, 10],
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-large"
) -> TestResult:
    """
    Esegue la valutazione completa del sistema RAG.
    
    Args:
        gold_queries: Dataset GOLD con ground truth
        retrieval_results: Risultati del retrieval
        generation_results: Risultati della generation
        configuration: Configurazione del sistema testato
        client: Client AzureOpenAI
        k_values: Valori di k per precision/recall
        llm_model: Modello LLM per valutazioni
        embedding_model: Modello per embeddings
        
    Returns:
        TestResult con tutti i risultati della valutazione
    """
    start_time = time.time()
    
    print("\n" + "="*70)
    print("AVVIO VALUTAZIONE SISTEMA RAG")
    print("="*70)
    
    # Mappa query_id -> oggetti
    gold_map = {q.query_id: q for q in gold_queries}
    retrieval_map = {r.query_id: r for r in retrieval_results}
    generation_map = {g.query_id: g for g in generation_results}
    
    # Risultati per query
    per_query_retrieval_metrics = []
    per_query_generation_metrics = []
    per_query_details = []
    
    total_queries = len(gold_queries)
    
    for i, query in enumerate(gold_queries, 1):
        print(f"\n[{i}/{total_queries}] Valutazione query: {query.query_id}")
        
        query_details = {
            "query_id": query.query_id,
            "query_text": query.query_text
        }
        
        # Valutazione Retrieval
        if query.query_id in retrieval_map:
            print("  → Valutazione retrieval...")
            retrieval_result = retrieval_map[query.query_id]
            retrieval_metrics = evaluate_retrieval_for_query(
                query, retrieval_result, client, k_values, llm_model
            )
            per_query_retrieval_metrics.append(retrieval_metrics)
            query_details["retrieval_metrics"] = retrieval_metrics
            print(f"    ✓ P@5: {retrieval_metrics.get('precision_at_5', 0):.3f}, "
                  f"R@5: {retrieval_metrics.get('recall_at_5', 0):.3f}, "
                  f"LLM-Relevance: {retrieval_metrics.get('avg_chunk_relevance_top5', 0):.3f}")
        
        # Valutazione Generation
        if query.query_id in generation_map:
            print("  → Valutazione generation...")
            generation_result = generation_map[query.query_id]
            generation_metrics = evaluate_generation_for_query(
                query, generation_result, client, llm_model, embedding_model
            )
            per_query_generation_metrics.append(generation_metrics)
            query_details["generation_metrics"] = generation_metrics
            print(f"    ✓ Faithfulness: {generation_metrics.get('faithfulness', 0):.3f}, "
                  f"Relevancy: {generation_metrics.get('answer_relevancy', 0):.3f}, "
                  f"Semantic Sim: {generation_metrics.get('semantic_similarity', 0):.3f}")
        
        per_query_details.append(query_details)
    
    # Aggregazione metriche
    print("\n" + "="*70)
    print("AGGREGAZIONE RISULTATI")
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

def save_test_result(test_result: TestResult, output_dir: str = "evaluation_results"):
    """
    Salva il risultato del test in un file JSON.
    
    Args:
        test_result: Risultato del test da salvare
        output_dir: Directory dove salvare i risultati
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    filename = f"{test_result.test_id}.json"
    filepath = Path(output_dir) / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(asdict(test_result), f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Risultati salvati: {filepath}")


def load_test_result(filepath: str) -> TestResult:
    """
    Carica un risultato di test da file JSON.
    
    Args:
        filepath: Percorso al file JSON
        
    Returns:
        TestResult caricato
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return TestResult(**data)


def load_all_test_results(results_dir: str = "evaluation_results") -> List[TestResult]:
    """
    Carica tutti i risultati di test da una directory.
    
    Args:
        results_dir: Directory con i risultati
        
    Returns:
        Lista di TestResult
    """
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        return results
    
    for json_file in results_path.glob("*.json"):
        results.append(load_test_result(str(json_file)))
    
    return results


def print_test_result(test_result: TestResult):
    """
    Stampa un report formattato del risultato di un test.
    
    Args:
        test_result: Risultato del test
    """
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


def compare_test_results(test_results: List[TestResult]):
    """
    Genera una tabella comparativa di più risultati di test.
    
    Args:
        test_results: Lista di TestResult da confrontare
    """
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


# ==================== FUNZIONI DI UTILITÀ ====================

def create_sample_results_for_testing():
    """
    Crea risultati di esempio per testare la pipeline (placeholder).
    DA SOSTITUIRE con chiamate al tuo sistema RAG reale.
    
    Returns:
        Tuple di (retrieval_results, generation_results)
    """
    retrieval_results = [
        RetrievalResult(
            query_id="q001",
            retrieved_chunk_ids=["doc_hr_manual_chunk_045", "doc_hr_manual_chunk_046", "doc_hr_manual_chunk_001"],
            retrieved_chunk_texts=[
                "Per richiedere le ferie, compilare il modulo FR-01 almeno 15 giorni prima.",
                "Il modulo deve essere approvato dal responsabile diretto tramite HR Portal.",
                "Le politiche aziendali sono disponibili nel manuale HR."
            ],
            retrieval_time=0.25
        )
    ]
    
    generation_results = [
        GenerationResult(
            query_id="q001",
            generated_answer="Per richiedere le ferie aziendali, è necessario compilare il modulo FR-01 con almeno 15 giorni di anticipo e ottenere l'approvazione del proprio responsabile attraverso il sistema HR Portal.",
            context_chunks=[
                "Per richiedere le ferie, compilare il modulo FR-01 almeno 15 giorni prima.",
                "Il modulo deve essere approvato dal responsabile diretto tramite HR Portal."
            ],
            generation_time=1.2
        )
    ]
    
    return retrieval_results, generation_results


# ==================== MAIN ====================

def main():
    """
    Esempio di utilizzo della pipeline di valutazione.
    """
    print("="*70)
    print("PIPELINE DI VALUTAZIONE SISTEMA RAG")
    print("Implementazione con LLM-as-judge e approccio funzionale")
    print("="*70)
    
    # 1. Configurazione
    print("\n[1/5] Configurazione AzureOpenAI client...")
    client = load_openai_client()  # Usa OPENAI_API_KEY da env
    
    # 2. Caricamento dataset GOLD
    print("\n[2/5] Caricamento dataset GOLD...")
    # gold_queries = load_gold_dataset("path/to/your/gold_dataset.json")
    
    # Per testing, uso esempio
    gold_queries = [
        EvaluationQuery(
            query_id="q001",
            query_text="Qual è la procedura per richiedere le ferie aziendali?",
            relevant_chunk_ids=["doc_hr_manual_chunk_045", "doc_hr_manual_chunk_046"],
            reference_answer="Per richiedere le ferie, compilare FR-01 almeno 15 giorni prima con approvazione del responsabile."
        )
    ]
    
    # 3. Ottenimento risultati dal tuo sistema RAG
    print("\n[3/5] Ottenimento risultati dal sistema RAG...")
    # ⚠️ SOSTITUISCI con chiamate al tuo sistema RAG
    retrieval_results, generation_results = create_sample_results_for_testing()
    
    # 4. Definizione configurazione del test
    print("\n[4/5] Configurazione del test...")
    configuration = {
        "name": "Test Baseline",
        "embedding_model": "text-embedding-3-large",
        "llm_model": "gpt-4o-mini",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "top_k": 5
    }
    
    # 5. Esecuzione valutazione
    print("\n[5/5] Esecuzione valutazione...")
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
    
    # Stampa risultati
    print_test_result(test_result)
    
    # Salvataggio
    save_test_result(test_result, "evaluation_results")
    
    print("\n✅ Esempio completato!")
    print("\nPer confrontare più configurazioni:")
    print("1. Modifica la configurazione (es. chunk_size=1024)")
    print("2. Esegui un nuovo test")
    print("3. Usa compare_test_results() per confrontare i risultati salvati")


if __name__ == "__main__":
    main()
