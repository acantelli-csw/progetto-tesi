"""
PIPELINE UNIFICATA DI VALUTAZIONE SISTEMA RAG
==============================================

Pipeline completa per la valutazione quantitativa di sistemi RAG con:
- Integrazione diretta con il sistema RAG esistente (search.py, llm.py)
- Metriche di retrieval (Precision@k, Recall@k standard e normalizzata, LLM-as-judge)
- Metriche di generation (Faithfulness, Answer Relevancy, Semantic Similarity)
- Supporto per diverse strategie: Semantic, Keyword (BM25), Hybrid, Multi-stage
- Supporto query negative: no_answer, correction, clarification

Tesi Magistrale in Ingegneria Informatica e dell'IA
Autore: Andrea Cantelli
Data: Gennaio 2026
"""

import os
import json
import time
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict, field
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
    SEMANTIC   = "semantic"    # Vector similarity
    KEYWORD    = "keyword"     # BM25
    HYBRID     = "hybrid"      # Combinazione fissa
    MULTISTAGE = "multistage"  # Pipeline multi-stage con tool selection adattivo

class ExpectedBehavior:
    """Comportamenti attesi per le query negative."""
    NO_ANSWER     = "no_answer"     # Il sistema deve dichiarare assenza di informazioni
    CORRECTION    = "correction"    # Il sistema deve correggere un dato errato nella query
    CLARIFICATION = "clarification" # Il sistema deve disambiguare prima di rispondere
    # Valore implicito per le query positive (nessun campo expected_behavior nel JSON)
    POSITIVE      = None

GOLD_DATASET_PATH = "C:/Users/ACantelli/OneDrive - centrosoftware.com/Documenti/GitHub/progetto-tesi/main/evaluation/gold_dataset.json"

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL_1 = os.getenv("EMBEDDING_MODEL_1")
EMBEDDING_URL_1   = os.getenv("EMBEDDING_URL_1")
EMBEDDING_VERSION_1 = os.getenv("EMBEDDING_VERSION_1")

EMBEDDING_MODEL_2 = os.getenv("EMBEDDING_MODEL_2")
EMBEDDING_URL_2   = os.getenv("EMBEDDING_URL_2")
EMBEDDING_VERSION_2 = os.getenv("EMBEDDING_VERSION_2")

LLM_MODEL   = os.getenv("LLM_MODEL")
LLM_URL     = os.getenv("LLM_URL")
LLM_VERSION = os.getenv("LLM_VERSION")


# ==================== STRUTTURE DATI ====================

@dataclass
class EvaluationQuery:
    """Query del dataset GOLD con ground truth."""
    query_id:           str
    query_text:         str
    relevant_chunk_ids: List[str]
    relevant_doc_ids:   List[str]           = field(default_factory=list)
    reference_answer:   Optional[str]       = None
    # Campi specifici query negative (None = query positiva)
    expected_behavior:  Optional[str]       = None
    negative_reason:    Optional[str]       = None

    @property
    def is_negative(self) -> bool:
        return self.expected_behavior is not None

@dataclass
class RetrievalResult:
    """Risultato del retrieval per una query."""
    query_id:              str
    retrieved_chunk_ids:   List[str]
    retrieved_chunk_texts: List[str]
    retrieval_time:        float
    tool_decision:         Optional[Dict] = None  # Solo per MULTISTAGE

@dataclass
class GenerationResult:
    """Risultato della generation per una query."""
    query_id:         str
    generated_answer: str
    context_chunks:   List[str]
    generation_time:  float

@dataclass
class NegativeEvaluationResult:
    """Risultato della valutazione per una query negativa."""
    query_id:          str
    expected_behavior: str
    behavior_score:    float   # 0.0 = comportamento errato, 1.0 = comportamento corretto
    behavior_label:    str     # Etichetta leggibile dell'esito
    llm_reasoning:     str     # Spiegazione del giudice LLM
    generation_time:   float

@dataclass
class TestResult:
    """Risultato completo di un test con una specifica configurazione."""
    test_id:                str
    timestamp:              str
    configuration:          Dict
    retrieval_metrics:      Dict[str, float]
    generation_metrics:     Dict[str, float]
    negative_eval_metrics:  Dict[str, float]
    per_query_details:      List[Dict]
    total_evaluation_time:  float


# ==================== FUNZIONI DI UTILITÀ ====================

def load_openai_client(api_key: Optional[str] = None) -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.getenv("LLM_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("LLM_VERSION")
    )

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def load_gold_dataset(filepath: str) -> List[EvaluationQuery]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    queries = []
    for item in data:
        query = EvaluationQuery(
            query_id           = item['query_id'],
            query_text         = item['query_text'],
            relevant_chunk_ids = item.get('relevant_chunk_ids', []),
            relevant_doc_ids   = item.get('relevant_doc_ids', []),
            reference_answer   = item.get('reference_answer'),
            expected_behavior  = item.get('expected_behavior'),
            negative_reason    = item.get('negative_reason')
        )
        queries.append(query)

    positive = sum(1 for q in queries if not q.is_negative)
    negative = sum(1 for q in queries if q.is_negative)
    print(f"  Dataset caricato: {len(queries)} query totali ({positive} positive, {negative} negative)")
    return queries


# ==================== METRICHE DI RETRIEVAL ====================

def calculate_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids:  List[str],
    k:             int
) -> float:
    """P@k = |{rilevanti} ∩ {top-k recuperati}| / k"""
    if k == 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for doc_id in top_k if doc_id in relevant_ids) / k

def calculate_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids:  List[str],
    k:             int
) -> float:
    """R@k = |{rilevanti} ∩ {top-k recuperati}| / |rilevanti|"""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for doc_id in top_k if doc_id in relevant_ids) / len(relevant_ids)

def calculate_recall_at_k_normalized(
    retrieved_ids: List[str],
    relevant_ids:  List[str],
    k:             int
) -> float:
    """
    R@k normalizzata = |{rilevanti} ∩ {top-k recuperati}| / min(k, |rilevanti|)

    Evita la penalizzazione strutturale per query con molti chunk rilevanti
    (es. una query con 11 chunk rilevanti non può superare R@5 = 45% con la
    formula standard, anche con un retriever perfetto).
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for doc_id in top_k if doc_id in relevant_ids) / min(k, len(relevant_ids))

def evaluate_chunk_relevance_with_llm(
    query:      str,
    chunk_text: str,
    client:     AzureOpenAI,
    model:      str = "gpt-4o-mini"
) -> float:
    """
    Valuta la rilevanza di un chunk usando una rubrica discreta a 3 livelli
    (0/1/2) per ridurre la varianza rispetto a scale continue.
    Il modello produce prima un ragionamento esplicito (chain-of-thought),
    poi emette il voto finale — questo aumenta la coerenza con temperatura 0.
    """
    prompt = f"""Sei un valutatore esperto di sistemi di recupero documenti.

Query utente: "{query}"

Chunk di documento:
\"\"\"
{chunk_text}
\"\"\"

Valuta la rilevanza del chunk rispetto alla query seguendo questa rubrica:
- 0 = Il chunk non contiene informazioni utili per rispondere alla query
- 1 = Il chunk contiene informazioni parzialmente utili o solo indirettamente correlate
- 2 = Il chunk contiene informazioni direttamente utili per rispondere alla query

Ragiona brevemente (1-2 frasi), poi emetti il voto.

Formato risposta (rispetta esattamente):
Ragionamento: <testo>
Voto: <0, 1 o 2>"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=80
        )
        text = response.choices[0].message.content.strip()
        # Estrae il voto dall'ultima riga
        for line in reversed(text.splitlines()):
            if line.startswith("Voto:"):
                score = int(line.split(":")[1].strip())
                return max(0.0, min(1.0, score / 2.0))  # normalizza a [0,1]
        return 0.0
    except Exception as e:
        print(f"  Errore valutazione LLM chunk relevance: {e}")
        return 0.0

def calculate_average_chunk_relevance(
    query:             str,
    retrieved_chunks:  List[str],
    client:            AzureOpenAI,
    model:             str = "gpt-4o-mini",
    top_k:             Optional[int] = None
) -> float:
    chunks_to_eval = retrieved_chunks[:top_k] if top_k else retrieved_chunks
    if not chunks_to_eval:
        return 0.0
    scores = [evaluate_chunk_relevance_with_llm(query, c, client, model) for c in chunks_to_eval]
    return float(np.mean(scores))

def evaluate_retrieval_for_query(
    query:            EvaluationQuery,
    retrieval_result: RetrievalResult,
    client:           AzureOpenAI,
    k_values:         List[int] = [1, 3, 5, 10],
    llm_model:        str = "gpt-4o-mini"
) -> Dict[str, float]:
    """
    Calcola le metriche di retrieval per una singola query positiva.
    Per ogni k calcola:
      - precision_at_k       (chunk-level, usa relevant_chunk_ids)
      - recall_at_k          (chunk-level, standard)
      - recall_at_k_norm     (chunk-level, normalizzata per min(k, |R|))
      - doc_recall_at_k      (document-level, usa relevant_doc_ids — invariante al chunking)
      - avg_chunk_relevance  (LLM-as-judge con rubrica discreta)
    """
    metrics = {}

    chunk_ids = retrieval_result.retrieved_chunk_ids
    # Per doc-level: estrae il NumRI dai chunk recuperati
    retrieved_doc_ids = list(dict.fromkeys(cid.split("_")[0] for cid in chunk_ids))

    for k in k_values:
        # --- Chunk-level ---
        metrics[f"precision_at_{k}"] = calculate_precision_at_k(
            chunk_ids, query.relevant_chunk_ids, k
        )
        metrics[f"recall_at_{k}"] = calculate_recall_at_k(
            chunk_ids, query.relevant_chunk_ids, k
        )
        metrics[f"recall_at_{k}_norm"] = calculate_recall_at_k_normalized(
            chunk_ids, query.relevant_chunk_ids, k
        )
        # --- Document-level (utile per confronto tra chunking diversi) ---
        if query.relevant_doc_ids:
            metrics[f"doc_recall_at_{k}"] = calculate_recall_at_k(
                retrieved_doc_ids, query.relevant_doc_ids, k
            )

        # --- LLM-as-judge chunk relevance ---
        metrics[f"avg_chunk_relevance_top{k}"] = calculate_average_chunk_relevance(
            query.query_text,
            retrieval_result.retrieved_chunk_texts,
            client, llm_model, top_k=k
        )

    metrics["retrieval_time"] = retrieval_result.retrieval_time
    return metrics


# ==================== METRICHE DI GENERATION (QUERY POSITIVE) ====================

def evaluate_faithfulness_with_llm(
    context_chunks:   List[str],
    generated_answer: str,
    client:           AzureOpenAI,
    model:            str = "gpt-4o-mini"
) -> float:
    """
    Valuta la fedeltà della risposta al contesto con rubrica discreta a 3 livelli.
    """
    context_text = "\n\n".join(context_chunks)

    prompt = f"""Sei un valutatore esperto di sistemi RAG.

Contesto (estratto dai documenti):
\"\"\"
{context_text}
\"\"\"

Risposta generata:
\"\"\"
{generated_answer}
\"\"\"

Valuta la fedeltà della risposta al contesto seguendo questa rubrica:
- 0 = La risposta contiene informazioni non presenti nel contesto (allucinazioni)
- 1 = La risposta è parzialmente fedele: alcune informazioni sono nel contesto, altre no
- 2 = La risposta è completamente fedele: ogni informazione è supportata dal contesto

Ragiona brevemente (1-2 frasi), poi emetti il voto.

Formato risposta (rispetta esattamente):
Ragionamento: <testo>
Voto: <0, 1 o 2>"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=80
        )
        text = response.choices[0].message.content.strip()
        for line in reversed(text.splitlines()):
            if line.startswith("Voto:"):
                score = int(line.split(":")[1].strip())
                return max(0.0, min(1.0, score / 2.0))
        return 0.0
    except Exception as e:
        print(f"  Errore valutazione faithfulness: {e}")
        return 0.0

def evaluate_answer_relevancy_with_llm(
    query:            str,
    generated_answer: str,
    client:           AzureOpenAI,
    model:            str = "gpt-4o-mini"
) -> float:
    """
    Valuta la pertinenza della risposta rispetto alla query con rubrica discreta.
    """
    prompt = f"""Sei un valutatore esperto di sistemi RAG.

Query utente: "{query}"

Risposta generata:
\"\"\"
{generated_answer}
\"\"\"

Valuta la pertinenza della risposta alla query seguendo questa rubrica:
- 0 = La risposta non risponde alla domanda posta
- 1 = La risposta risponde parzialmente o in modo indiretto alla domanda
- 2 = La risposta risponde direttamente e completamente alla domanda

Ragiona brevemente (1-2 frasi), poi emetti il voto.

Formato risposta (rispetta esattamente):
Ragionamento: <testo>
Voto: <0, 1 o 2>"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=80
        )
        text = response.choices[0].message.content.strip()
        for line in reversed(text.splitlines()):
            if line.startswith("Voto:"):
                score = int(line.split(":")[1].strip())
                return max(0.0, min(1.0, score / 2.0))
        return 0.0
    except Exception as e:
        print(f"  Errore valutazione answer relevancy: {e}")
        return 0.0

def calculate_semantic_similarity_embeddings(
    reference_answer: str,
    generated_answer: str,
) -> float:
    """Similarità coseno tra embedding della reference e della risposta generata."""
    try:
        return cosine_similarity(
            get_embedding(reference_answer),
            get_embedding(generated_answer)
        )
    except Exception as e:
        print(f"  Errore calcolo semantic similarity: {e}")
        return 0.0

def evaluate_generation_for_query(
    query:             EvaluationQuery,
    generation_result: GenerationResult,
    client:            AzureOpenAI,
    llm_model:         str = "gpt-4o-mini",
    embedding_model:   str = "text-embedding-3-large"
) -> Dict[str, float]:

    metrics = {}

    metrics["faithfulness"] = evaluate_faithfulness_with_llm(
        generation_result.context_chunks,
        generation_result.generated_answer,
        client, llm_model
    )
    metrics["answer_relevancy"] = evaluate_answer_relevancy_with_llm(
        query.query_text,
        generation_result.generated_answer,
        client, llm_model
    )
    if query.reference_answer:
        metrics["semantic_similarity"] = calculate_semantic_similarity_embeddings(
            query.reference_answer,
            generation_result.generated_answer
        )
    else:
        metrics["semantic_similarity"] = None

    metrics["generation_time"] = generation_result.generation_time
    return metrics


# ==================== VALUTAZIONE QUERY NEGATIVE ====================

def evaluate_no_answer_behavior(
    query:            EvaluationQuery,
    generated_answer: str,
    client:           AzureOpenAI,
    model:            str = "gpt-4o-mini"
) -> Tuple[float, str, str]:
    """
    Verifica se il sistema ha correttamente dichiarato di non avere informazioni
    sufficienti per rispondere, invece di generare contenuto inventato.

    Ritorna: (score 0.0/1.0, label, reasoning)
    """
    prompt = f"""Sei un valutatore esperto di sistemi RAG.

La seguente domanda riguarda un'informazione NON presente nel database documentale del sistema:
Query: "{query.query_text}"
Motivo per cui non dovrebbe rispondere: {query.negative_reason}

Risposta generata dal sistema:
\"\"\"
{generated_answer}
\"\"\"

Valuta se il sistema ha risposto correttamente seguendo questa rubrica:
- 0 = Il sistema ha risposto come se avesse le informazioni (allucinazione o risposta generica non supportata)
- 1 = Il sistema ha esplicitamente dichiarato di non avere informazioni sufficienti per rispondere

Ragiona brevemente (1-2 frasi), poi emetti il voto.

Formato risposta (rispetta esattamente):
Ragionamento: <testo>
Voto: <0 o 1>"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100
        )
        text = response.choices[0].message.content.strip()
        reasoning = ""
        score = 0.0
        for line in text.splitlines():
            if line.startswith("Ragionamento:"):
                reasoning = line.split(":", 1)[1].strip()
            if line.startswith("Voto:"):
                score = float(line.split(":")[1].strip())
        label = "CORRETTO (ha dichiarato assenza info)" if score == 1.0 else "ERRATO (ha risposto senza basi)"
        return score, label, reasoning
    except Exception as e:
        print(f"  Errore valutazione no_answer: {e}")
        return 0.0, "ERRORE", str(e)

def evaluate_correction_behavior(
    query:            EvaluationQuery,
    generated_answer: str,
    client:           AzureOpenAI,
    model:            str = "gpt-4o-mini"
) -> Tuple[float, str, str]:
    """
    Verifica se il sistema ha identificato e corretto il dato errato presente
    nella query, invece di confermarlo.

    Ritorna: (score 0.0/1.0, label, reasoning)
    """
    prompt = f"""Sei un valutatore esperto di sistemi RAG.

La seguente domanda contiene un dato errato che il sistema dovrebbe correggere:
Query: "{query.query_text}"
Dato errato e correzione attesa: {query.negative_reason}

Risposta generata dal sistema:
\"\"\"
{generated_answer}
\"\"\"

Valuta se il sistema ha risposto correttamente seguendo questa rubrica:
- 0 = Il sistema ha confermato o ignorato il dato errato senza correggerlo
- 1 = Il sistema ha esplicitamente identificato il dato errato e fornito la correzione corretta

Ragiona brevemente (1-2 frasi), poi emetti il voto.

Formato risposta (rispetta esattamente):
Ragionamento: <testo>
Voto: <0 o 1>"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100
        )
        text = response.choices[0].message.content.strip()
        reasoning = ""
        score = 0.0
        for line in text.splitlines():
            if line.startswith("Ragionamento:"):
                reasoning = line.split(":", 1)[1].strip()
            if line.startswith("Voto:"):
                score = float(line.split(":")[1].strip())
        label = "CORRETTO (ha corretto il dato errato)" if score == 1.0 else "ERRATO (ha confermato o ignorato il dato)"
        return score, label, reasoning
    except Exception as e:
        print(f"  Errore valutazione correction: {e}")
        return 0.0, "ERRORE", str(e)

def evaluate_clarification_behavior(
    query:            EvaluationQuery,
    generated_answer: str,
    client:           AzureOpenAI,
    model:            str = "gpt-4o-mini"
) -> Tuple[float, str, str]:
    """
    Verifica se il sistema ha gestito correttamente una query ambigua,
    chiedendo chiarimenti o coprendo esplicitamente le diverse interpretazioni.

    Ritorna: (score 0.0/1.0, label, reasoning)
    """
    prompt = f"""Sei un valutatore esperto di sistemi RAG.

La seguente domanda è ambigua e ammette interpretazioni multiple:
Query: "{query.query_text}"
Motivo dell'ambiguità: {query.negative_reason}

Risposta generata dal sistema:
\"\"\"
{generated_answer}
\"\"\"

Valuta se il sistema ha gestito l'ambiguità correttamente seguendo questa rubrica:
- 0 = Il sistema ha risposto a una sola interpretazione senza riconoscere l'ambiguità
- 1 = Il sistema ha riconosciuto l'ambiguità e ha chiesto chiarimenti OPPURE ha coperto esplicitamente le diverse interpretazioni possibili

Ragiona brevemente (1-2 frasi), poi emetti il voto.

Formato risposta (rispetta esattamente):
Ragionamento: <testo>
Voto: <0 o 1>"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100
        )
        text = response.choices[0].message.content.strip()
        reasoning = ""
        score = 0.0
        for line in text.splitlines():
            if line.startswith("Ragionamento:"):
                reasoning = line.split(":", 1)[1].strip()
            if line.startswith("Voto:"):
                score = float(line.split(":")[1].strip())
        label = "CORRETTO (ha gestito l'ambiguità)" if score == 1.0 else "ERRATO (ha ignorato l'ambiguità)"
        return score, label, reasoning
    except Exception as e:
        print(f"  Errore valutazione clarification: {e}")
        return 0.0, "ERRORE", str(e)

def evaluate_negative_query(
    query:             EvaluationQuery,
    generation_result: GenerationResult,
    client:            AzureOpenAI,
    model:             str = "gpt-4o-mini"
) -> NegativeEvaluationResult:
    """
    Router principale per la valutazione delle query negative.
    Seleziona la funzione di valutazione corretta in base a expected_behavior.
    """
    start = time.time()

    if query.expected_behavior == ExpectedBehavior.NO_ANSWER:
        score, label, reasoning = evaluate_no_answer_behavior(
            query, generation_result.generated_answer, client, model
        )
    elif query.expected_behavior == ExpectedBehavior.CORRECTION:
        score, label, reasoning = evaluate_correction_behavior(
            query, generation_result.generated_answer, client, model
        )
    elif query.expected_behavior == ExpectedBehavior.CLARIFICATION:
        score, label, reasoning = evaluate_clarification_behavior(
            query, generation_result.generated_answer, client, model
        )
    else:
        score, label, reasoning = 0.0, "TIPO NON GESTITO", f"expected_behavior sconosciuto: {query.expected_behavior}"

    return NegativeEvaluationResult(
        query_id          = query.query_id,
        expected_behavior = query.expected_behavior,
        behavior_score    = score,
        behavior_label    = label,
        llm_reasoning     = reasoning,
        generation_time   = time.time() - start
    )

def aggregate_negative_metrics(results: List[NegativeEvaluationResult]) -> Dict[str, float]:
    """
    Calcola metriche aggregate per le query negative, suddivise per tipo
    di expected_behavior e aggregate complessivamente.
    """
    if not results:
        return {}

    by_type: Dict[str, List[float]] = {}
    for r in results:
        by_type.setdefault(r.expected_behavior, []).append(r.behavior_score)

    metrics = {}
    all_scores = []
    for behavior, scores in by_type.items():
        metrics[f"robustness_{behavior}"] = float(np.mean(scores))
        metrics[f"count_{behavior}"]      = len(scores)
        all_scores.extend(scores)

    metrics["robustness_overall"] = float(np.mean(all_scores))
    metrics["count_total"]        = len(all_scores)
    return metrics


# ==================== AGGREGAZIONE METRICHE POSITIVE ====================

def aggregate_retrieval_metrics(per_query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_query_metrics:
        return {}
    metric_keys = set(k for m in per_query_metrics for k in m.keys())
    aggregated = {}
    for key in metric_keys:
        values = [m[key] for m in per_query_metrics if key in m and m[key] is not None]
        if values:
            aggregated[f"avg_{key}"] = float(np.mean(values))
    return aggregated

def aggregate_generation_metrics(per_query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_query_metrics:
        return {}
    metric_keys = set(k for m in per_query_metrics for k in m.keys())
    aggregated = {}
    for key in metric_keys:
        values = [m[key] for m in per_query_metrics if key in m and m[key] is not None]
        if values:
            aggregated[f"avg_{key}"] = float(np.mean(values))
    return aggregated


# ==================== INTEGRAZIONE CON SISTEMA RAG ====================

def run_retrieval_with_semantic_search(
    queries: List[EvaluationQuery],
    top_k:   int = 10
) -> List[RetrievalResult]:

    retrieval_results = []
    print(f"\n{'='*70}")
    print(f"RETRIEVAL SEMANTICO: {len(queries)} query")
    print(f"{'='*70}\n")

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query.query_text[:60]}...")
        start_time = time.time()
        try:
            docs = search.semantic_search(prompt=query.query_text, top_n=top_k)
            retrieved_chunk_ids   = [f"{d['numero']}_{d['progressivo']}" for d in docs]
            retrieved_chunk_texts = [d['content'] for d in docs]
            print(f"  ✓ {len(retrieved_chunk_ids)} chunk recuperati")
        except Exception as e:
            print(f"  ✗ Errore: {e}")
            retrieved_chunk_ids, retrieved_chunk_texts = [], []

        retrieval_results.append(RetrievalResult(
            query_id              = query.query_id,
            retrieved_chunk_ids   = retrieved_chunk_ids,
            retrieved_chunk_texts = retrieved_chunk_texts,
            retrieval_time        = time.time() - start_time
        ))

    print(f"\n{'='*70}\n")
    return retrieval_results

def run_retrieval_with_keyword_search(
    queries: List[EvaluationQuery],
    top_k:   int = 10
) -> List[RetrievalResult]:

    retrieval_results = []
    print(f"\n{'='*70}")
    print(f"RETRIEVAL BM25: {len(queries)} query")
    print(f"{'='*70}\n")

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query.query_text[:60]}...")
        start_time = time.time()
        try:
            docs = search.keyword_search(prompt=query.query_text, top_n=top_k, language='italian')
            retrieved_chunk_ids   = [f"{d['numero']}_{d['progressivo']}" for d in docs]
            retrieved_chunk_texts = [d['content'] for d in docs]
            print(f"  ✓ {len(retrieved_chunk_ids)} chunk recuperati")
        except Exception as e:
            print(f"  ✗ Errore: {e}")
            retrieved_chunk_ids, retrieved_chunk_texts = [], []

        retrieval_results.append(RetrievalResult(
            query_id              = query.query_id,
            retrieved_chunk_ids   = retrieved_chunk_ids,
            retrieved_chunk_texts = retrieved_chunk_texts,
            retrieval_time        = time.time() - start_time
        ))

    print(f"\n{'='*70}\n")
    return retrieval_results

def run_retrieval_hybrid(
    queries:          List[EvaluationQuery],
    top_k:            int   = 10,
    semantic_weight:  float = 0.7
) -> List[RetrievalResult]:

    retrieval_results = []
    keyword_weight = 1.0 - semantic_weight
    print(f"\n{'='*70}")
    print(f"RETRIEVAL IBRIDO: {len(queries)} query  (semantic={semantic_weight}, keyword={keyword_weight})")
    print(f"{'='*70}\n")

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query.query_text[:60]}...")
        start_time = time.time()
        try:
            semantic_docs = search.semantic_search(query.query_text, top_n=top_k * 2)
            keyword_docs  = search.keyword_search(query.query_text, top_n=top_k * 2)

            combined = {}
            for doc in semantic_docs:
                cid = f"{doc['numero']}_{doc['progressivo']}"
                combined[cid] = {'doc': doc, 'score': doc['similarity'] * semantic_weight}

            for doc in keyword_docs:
                cid = f"{doc['numero']}_{doc['progressivo']}"
                norm_score = doc['score'] / (doc['score'] + 1)
                if cid in combined:
                    combined[cid]['score'] += norm_score * keyword_weight
                else:
                    combined[cid] = {'doc': doc, 'score': norm_score * keyword_weight}

            sorted_results = sorted(combined.items(), key=lambda x: x[1]['score'], reverse=True)[:top_k]
            retrieved_chunk_ids   = [cid for cid, _ in sorted_results]
            retrieved_chunk_texts = [data['doc']['content'] for _, data in sorted_results]
            print(f"  ✓ {len(retrieved_chunk_ids)} chunk recuperati")
        except Exception as e:
            print(f"  ✗ Errore: {e}")
            retrieved_chunk_ids, retrieved_chunk_texts = [], []

        retrieval_results.append(RetrievalResult(
            query_id              = query.query_id,
            retrieved_chunk_ids   = retrieved_chunk_ids,
            retrieved_chunk_texts = retrieved_chunk_texts,
            retrieval_time        = time.time() - start_time
        ))

    print(f"\n{'='*70}\n")
    return retrieval_results

def run_retrieval_with_multistage(
    queries: List[EvaluationQuery],
    top_k:   int = 10
) -> List[RetrievalResult]:
    """
    Retrieval adattivo: chiama decide_tools() per scegliere il retriever
    per ciascuna query, tracciando le decisioni per calcolare pct_avoided_hybrid.
    """
    retrieval_results = []
    print(f"\n{'='*70}")
    print(f"RETRIEVAL MULTI-STAGE (adattivo): {len(queries)} query")
    print(f"{'='*70}\n")

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query.query_text[:60]}...")
        start_time = time.time()
        try:
            tool_decision = your_llm.decide_tools(query.query_text)
            use_semantic  = tool_decision.get("use_semantic", False)
            use_keyword   = tool_decision.get("use_keyword",  False)

            all_docs = []
            if use_semantic:
                sem_docs = search.semantic_search(query.query_text, top_n=top_k)
                for d in sem_docs:
                    d["retrieval_sources"] = ["semantic"]
                all_docs += sem_docs
            if use_keyword:
                kw_docs = search.keyword_search(query.query_text, top_n=top_k)
                for d in kw_docs:
                    d["retrieval_sources"] = ["keyword"]
                all_docs += kw_docs

            # Deduplicazione con merge sorgenti
            total_before = len(all_docs)
            seen = {}
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

            co_count = total_before - len(all_docs)
            if total_before > 0:
                print(f"  → Co-retrieval: {co_count}/{total_before} chunk ({co_count/total_before*100:.1f}%)")

            retrieved_chunk_ids   = [f"{d['numero']}_{d['progressivo']}" for d in all_docs]
            retrieved_chunk_texts = [d['content'] for d in all_docs]

            mode = ("HYBRID" if use_semantic and use_keyword
                    else "SEMANTIC" if use_semantic
                    else "KEYWORD" if use_keyword
                    else "NESSUNA")
            print(f"  → Tool selection: {mode}  |  {tool_decision.get('reason', '')[:70]}")
            print(f"  ✓ {len(retrieved_chunk_ids)} chunk recuperati")

        except Exception as e:
            print(f"  ✗ Errore: {e}")
            retrieved_chunk_ids, retrieved_chunk_texts = [], []
            tool_decision = {"use_semantic": False, "use_keyword": False, "reason": f"Errore: {e}"}

        retrieval_results.append(RetrievalResult(
            query_id              = query.query_id,
            retrieved_chunk_ids   = retrieved_chunk_ids,
            retrieved_chunk_texts = retrieved_chunk_texts,
            retrieval_time        = time.time() - start_time,
            tool_decision         = tool_decision
        ))

    print(f"\n{'='*70}\n")
    return retrieval_results

def aggregate_tool_selection_stats(retrieval_results: List[RetrievalResult]) -> Dict:
    decisions = [r.tool_decision for r in retrieval_results if r.tool_decision is not None]
    if not decisions:
        return {}

    total          = len(decisions)
    semantic_only  = sum(1 for d in decisions if     d.get("use_semantic") and not d.get("use_keyword"))
    keyword_only   = sum(1 for d in decisions if     d.get("use_keyword")  and not d.get("use_semantic"))
    both           = sum(1 for d in decisions if     d.get("use_semantic") and     d.get("use_keyword"))
    none_selected  = sum(1 for d in decisions if not d.get("use_semantic") and not d.get("use_keyword"))

    return {
        "total_queries":        total,
        "semantic_only_count":  semantic_only,
        "keyword_only_count":   keyword_only,
        "both_count":           both,
        "none_count":           none_selected,
        "pct_semantic_only":    round(semantic_only / total * 100, 1),
        "pct_keyword_only":     round(keyword_only  / total * 100, 1),
        "pct_both":             round(both          / total * 100, 1),
        "pct_none":             round(none_selected / total * 100, 1),
        "pct_avoided_hybrid":   round((semantic_only + keyword_only + none_selected) / total * 100, 1),
        "per_query_decisions": [
            {
                "query_id":    r.query_id,
                "use_semantic": r.tool_decision.get("use_semantic"),
                "use_keyword":  r.tool_decision.get("use_keyword"),
                "mode": (
                    "both"          if r.tool_decision.get("use_semantic") and r.tool_decision.get("use_keyword")
                    else "semantic_only" if r.tool_decision.get("use_semantic")
                    else "keyword_only"  if r.tool_decision.get("use_keyword")
                    else "none"
                ),
                "reason": r.tool_decision.get("reason", "")
            }
            for r in retrieval_results if r.tool_decision is not None
        ]
    }


# ==================== GENERATION ====================

def run_generation_with_llm(
    queries:          List[EvaluationQuery],
    retrieval_results: List[RetrievalResult],
    llm_model:        str = "gpt-4o-mini"
) -> List[GenerationResult]:

    generation_results = []
    print(f"\n{'='*70}")
    print(f"GENERATION: {len(queries)} query")
    print(f"{'='*70}\n")

    for i, (query, retrieval_result) in enumerate(zip(queries, retrieval_results), 1):
        print(f"[{i}/{len(queries)}] {query.query_text[:60]}...")
        start_time = time.time()

        chunk_ids   = retrieval_result.retrieved_chunk_ids
        chunk_texts = retrieval_result.retrieved_chunk_texts

        try:
            fake_docs = []
            for idx, (chunk_id, chunk_text) in enumerate(zip(chunk_ids, chunk_texts)):
                try:
                    num_ri, progressivo = chunk_id.split('_')
                except ValueError:
                    num_ri, progressivo = "UNKNOWN", str(idx)
                fake_docs.append({
                    'numero':     num_ri,
                    'progressivo': int(progressivo),
                    'titolo':     f"Documento {idx + 1}",
                    'autore':     "Sistema",
                    'cliente':    "Test",
                    'content':    chunk_text
                })

            answer = your_llm.generate_final_answer(
                user_prompt   = query.query_text,
                selected_docs = fake_docs,
                chat_history  = []
            )

            import types
            if isinstance(answer, types.GeneratorType):
                answer = ''.join(answer)
                print(f"  ✓ Risposta generata (stream): {len(answer)} char")
            elif isinstance(answer, str):
                print(f"  ✓ Risposta generata: {len(answer)} char")
            else:
                answer = str(answer)

            if not answer or not answer.strip():
                answer = "[RISPOSTA VUOTA]"

        except Exception as e:
            print(f"  ✗ Errore: {e}")
            import traceback; traceback.print_exc()
            answer = f"[ERRORE] {str(e)}"

        generation_results.append(GenerationResult(
            query_id         = query.query_id,
            generated_answer = answer,
            context_chunks   = chunk_texts,
            generation_time  = time.time() - start_time
        ))

    print(f"\n{'='*70}\n")
    return generation_results


# ==================== VALUTAZIONE COMPLETA ====================

def run_full_evaluation(
    gold_queries:       List[EvaluationQuery],
    retrieval_results:  List[RetrievalResult],
    generation_results: List[GenerationResult],
    configuration:      Dict,
    client:             AzureOpenAI,
    k_values:           List[int] = [5, 10],
    llm_model:          str = "gpt-4o-mini",
    embedding_model:    str = "text-embedding-3-large"
) -> TestResult:

    start_time = time.time()
    print("\n" + "="*70)
    print("VALUTAZIONE METRICHE")
    print("="*70)

    per_query_retrieval_metrics  = []
    per_query_generation_metrics = []
    negative_eval_results        = []
    per_query_details            = []

    for i, (query, ret, gen) in enumerate(
        zip(gold_queries, retrieval_results, generation_results), 1
    ):
        print(f"\n[{i}/{len(gold_queries)}] {query.query_id}  {'[NEGATIVA]' if query.is_negative else ''}")

        if query.is_negative:
            # --- Valutazione query negativa ---
            # Le metriche di retrieval standard non si applicano (no chunk rilevanti attesi
            # per no_answer/clarification, oppure hanno senso parziale per correction).
            # Valutiamo solo il comportamento della generation.
            print(f"  - Comportamento atteso: {query.expected_behavior}")
            neg_result = evaluate_negative_query(query, gen, client, llm_model)
            negative_eval_results.append(neg_result)
            print(f"  → {neg_result.behavior_label}")

            per_query_details.append({
                "query_id":         query.query_id,
                "query_text":       query.query_text,
                "is_negative":      True,
                "expected_behavior": query.expected_behavior,
                "behavior_score":   neg_result.behavior_score,
                "behavior_label":   neg_result.behavior_label,
                "llm_reasoning":    neg_result.llm_reasoning,
                "generated_answer": gen.generated_answer,
            })

        else:
            # --- Valutazione query positiva ---
            print("  - Metriche retrieval...")
            ret_metrics = evaluate_retrieval_for_query(
                query, ret, client, k_values, llm_model
            )
            per_query_retrieval_metrics.append(ret_metrics)

            print("  - Metriche generation...")
            gen_metrics = evaluate_generation_for_query(
                query, gen, client, llm_model, embedding_model
            )
            per_query_generation_metrics.append(gen_metrics)

            per_query_details.append({
                "query_id":             query.query_id,
                "query_text":           query.query_text,
                "is_negative":          False,
                "retrieval_metrics":    ret_metrics,
                "generation_metrics":   gen_metrics,
                "generated_answer":     gen.generated_answer,
                "retrieved_chunk_count": len(ret.retrieved_chunk_ids),
            })

        print(f"  ✓ {query.query_id} completata")

    print("\n" + "="*70)
    print("AGGREGAZIONE METRICHE")
    print("="*70)

    aggregated_retrieval  = aggregate_retrieval_metrics(per_query_retrieval_metrics)
    aggregated_generation = aggregate_generation_metrics(per_query_generation_metrics)
    aggregated_negative   = aggregate_negative_metrics(negative_eval_results)

    return TestResult(
        test_id                = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        timestamp              = datetime.now().isoformat(),
        configuration          = configuration,
        retrieval_metrics      = aggregated_retrieval,
        generation_metrics     = aggregated_generation,
        negative_eval_metrics  = aggregated_negative,
        per_query_details      = per_query_details,
        total_evaluation_time  = time.time() - start_time
    )


# ==================== GESTIONE RISULTATI ====================

def save_test_result(test_result: TestResult, output_dir: str = "evaluation_results"):
    Path(output_dir).mkdir(exist_ok=True)
    filepath = Path(output_dir) / f"{test_result.test_id}.json"
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
    for k, v in test_result.configuration.items():
        if k != "tool_selection_stats":
            print(f"  • {k}: {v}")

    print("\n📊 METRICHE RETRIEVAL (query positive):")
    for metric, value in sorted(test_result.retrieval_metrics.items()):
        print(f"  • {metric}: {value:.4f}")

    print("\n📝 METRICHE GENERATION (query positive):")
    for metric, value in sorted(test_result.generation_metrics.items()):
        if value is not None:
            print(f"  • {metric}: {value:.4f}")

    print("\n🛡️  ROBUSTEZZA (query negative):")
    for metric, value in sorted(test_result.negative_eval_metrics.items()):
        if isinstance(value, float):
            print(f"  • {metric}: {value:.4f}")
        else:
            print(f"  • {metric}: {value}")

    print("\n" + "="*70)

def compare_test_results(test_results: List[TestResult]):
    if not test_results:
        return

    print("\n" + "="*130)
    print("CONFRONTO RISULTATI TEST")
    print("="*130)

    col = 38
    print(f"\n{'Configurazione':<{col}} {'P@5':>7} {'R@5':>7} {'R@5n':>7} {'LLM-Rel':>8} {'Faith':>7} {'Relev':>7} {'SemSim':>7} {'Robust':>8}")
    print("-"*130)

    for result in test_results:
        name    = result.configuration.get('name', result.test_id)[:col - 2]
        p5      = result.retrieval_metrics.get('avg_precision_at_5', 0.0)
        r5      = result.retrieval_metrics.get('avg_recall_at_5', 0.0)
        r5n     = result.retrieval_metrics.get('avg_recall_at_5_norm', 0.0)
        llmrel  = result.retrieval_metrics.get('avg_avg_chunk_relevance_top5', 0.0)
        faith   = result.generation_metrics.get('avg_faithfulness', 0.0)
        relev   = result.generation_metrics.get('avg_answer_relevancy', 0.0)
        semsim  = result.generation_metrics.get('avg_semantic_similarity') or 0.0
        robust  = result.negative_eval_metrics.get('robustness_overall', 0.0)

        print(f"{name:<{col}} {p5:>7.4f} {r5:>7.4f} {r5n:>7.4f} {llmrel:>8.4f} {faith:>7.4f} {relev:>7.4f} {semsim:>7.4f} {robust:>8.4f}")

    print("="*130)
    print("\nLegenda:")
    print("  P@5     = Precision@5 (chunk-level)")
    print("  R@5     = Recall@5 standard (chunk-level)")
    print("  R@5n    = Recall@5 normalizzata — denominatore min(5, |R|)")
    print("  LLM-Rel = LLM-as-judge chunk relevance (rubrica 0/1/2, media top-5)")
    print("  Faith   = Faithfulness (LLM-as-judge, rubrica 0/1/2)")
    print("  Relev   = Answer Relevancy (LLM-as-judge, rubrica 0/1/2)")
    print("  SemSim  = Semantic Similarity (cosine su embeddings)")
    print("  Robust  = Robustezza su query negative (0=fallisce, 1=gestisce correttamente)")
    print()


# ==================== TEST PRINCIPALE ====================

def run_test_with_your_pipeline(
    gold_dataset_path: str,
    configuration:     dict,
    search_strategy:   str = SearchStrategy.SEMANTIC,
    results_dir:       str = "evaluation_results"
):
    print("\n" + "="*70)
    print(f"TEST: {configuration.get('name', 'Unnamed')}")
    print(f"Search Strategy: {search_strategy}")
    print("="*70)

    print("\n[1/4] Caricamento dataset GOLD...")
    gold_queries = load_gold_dataset(gold_dataset_path)

    print(f"\n[2/4] Retrieval con strategia: {search_strategy}...")
    top_k = configuration.get('top_k', 10)

    if   search_strategy == SearchStrategy.SEMANTIC:
        retrieval_results = run_retrieval_with_semantic_search(gold_queries, top_k)
    elif search_strategy == SearchStrategy.KEYWORD:
        retrieval_results = run_retrieval_with_keyword_search(gold_queries, top_k)
    elif search_strategy == SearchStrategy.HYBRID:
        retrieval_results = run_retrieval_hybrid(
            gold_queries, top_k,
            semantic_weight=configuration.get('semantic_weight', 0.7)
        )
    elif search_strategy == SearchStrategy.MULTISTAGE:
        retrieval_results = run_retrieval_with_multistage(gold_queries, top_k)
    else:
        raise ValueError(f"Search strategy non valida: {search_strategy}")

    print("\n[3/4] Generation...")
    generation_results = run_generation_with_llm(
        gold_queries, retrieval_results,
        llm_model=configuration.get('llm_model', 'gpt-4-1')
    )

    print("\n[4/4] Valutazione metriche...")
    client = load_openai_client()

    test_result = run_full_evaluation(
        gold_queries       = gold_queries,
        retrieval_results  = retrieval_results,
        generation_results = generation_results,
        configuration      = configuration,
        client             = client,
        k_values           = [3, 5, 10],
        llm_model          = "gpt-4o-mini",
        embedding_model    = "text-embedding-3-large"
    )

    print_test_result(test_result)

    if search_strategy == SearchStrategy.MULTISTAGE:
        tool_stats = aggregate_tool_selection_stats(retrieval_results)
        print_tool_selection_stats(tool_stats)
        test_result.configuration["tool_selection_stats"] = tool_stats

    save_test_result(test_result, results_dir)
    return test_result


# ==================== ESEMPI DI TEST ====================

def print_tool_selection_stats(stats: Dict):
    if not stats:
        return
    print("\n" + "="*70)
    print("📊 STATISTICHE TOOL SELECTION (Multi-stage)")
    print("="*70)
    print(f"\n  Totale query analizzate : {stats['total_queries']}")
    print(f"\n  Semantic only           : {stats['semantic_only_count']:>3}  ({stats['pct_semantic_only']:>5.1f}%)")
    print(f"  Keyword only            : {stats['keyword_only_count']:>3}  ({stats['pct_keyword_only']:>5.1f}%)")
    print(f"  Entrambi (≈ hybrid)     : {stats['both_count']:>3}  ({stats['pct_both']:>5.1f}%)")
    print(f"  Nessuna ricerca         : {stats['none_count']:>3}  ({stats['pct_none']:>5.1f}%)")
    print(f"\n  → Risparmio vs hybrid fisso: {stats['pct_avoided_hybrid']:.1f}% delle query")
    print(f"    ha evitato di eseguire entrambi i retriever\n")
    print("  Dettaglio per query:")
    print(f"  {'Query ID':<12} {'Mode':<16} {'Reason'}")
    print("  " + "-"*66)
    for d in stats.get("per_query_decisions", []):
        reason_short = d['reason'][:45] + "..." if len(d['reason']) > 45 else d['reason']
        print(f"  {d['query_id']:<12} {d['mode']:<16} {reason_short}")
    print("="*70)


def esempio_test_semantic_search():
    return run_test_with_your_pipeline(
        gold_dataset_path=GOLD_DATASET_PATH,
        configuration={
            "name": "Semantic Search",
            "search_strategy": "semantic",
            "embedding_model": EMBEDDING_MODEL_1,
            "llm_model": "gpt-4.1",
            "top_k": 10
        },
        search_strategy=SearchStrategy.SEMANTIC
    )

def esempio_test_keyword_search():
    return run_test_with_your_pipeline(
        gold_dataset_path=GOLD_DATASET_PATH,
        configuration={
            "name": "Keyword Search BM25",
            "search_strategy": "keyword",
            "llm_model": "gpt-4.1",
            "top_k": 10
        },
        search_strategy=SearchStrategy.KEYWORD
    )

def esempio_test_hybrid_search():
    return run_test_with_your_pipeline(
        gold_dataset_path=GOLD_DATASET_PATH,
        configuration={
            "name": "Hybrid Search (50/50)",
            "search_strategy": "hybrid",
            "semantic_weight": 0.5,
            "embedding_model": EMBEDDING_MODEL_1,
            "llm_model": "gpt-4.1",
            "top_k": 10
        },
        search_strategy=SearchStrategy.HYBRID
    )

def esempio_test_multistage():
    return run_test_with_your_pipeline(
        gold_dataset_path=GOLD_DATASET_PATH,
        configuration={
            "name": "Multi-stage (adattivo)",
            "search_strategy": "multistage",
            "embedding_model": EMBEDDING_MODEL_1,
            "llm_model": "gpt-4.1",
            "top_k": 10
        },
        search_strategy=SearchStrategy.MULTISTAGE
    )

def esempio_confronto_strategie():
    print("\n" + "="*70)
    print("CONFRONTO STRATEGIE DI SEARCH")
    print("="*70)
    results = []
    print("\n[TEST 1/4] Semantic Search...")
    results.append(esempio_test_semantic_search())
    print("\n[TEST 2/4] Keyword Search (BM25)...")
    results.append(esempio_test_keyword_search())
    print("\n[TEST 3/4] Hybrid Search...")
    results.append(esempio_test_hybrid_search())
    print("\n[TEST 4/4] Multi-stage (adattivo)...")
    results.append(esempio_test_multistage())
    compare_test_results(results)


# ==================== MAIN ====================

def main():
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

    if   choice == "1": esempio_test_semantic_search()
    elif choice == "2": esempio_test_keyword_search()
    elif choice == "3": esempio_test_hybrid_search()
    elif choice == "4": esempio_test_multistage()
    elif choice == "5": esempio_confronto_strategie()
    else:               print("Selezione non valida")


if __name__ == "__main__":
    main()