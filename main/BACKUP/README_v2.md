# Pipeline di Valutazione Sistema RAG v2.0

> **Tesi Magistrale in Ingegneria Informatica e dell'Intelligenza Artificiale**  
> Framework funzionale per valutazione quantitativa di sistemi RAG con LLM-as-judge

## 📋 Panoramica

Pipeline di valutazione completa per sistemi Retrieval-Augmented Generation (RAG) che utilizza:
- **Approccio funzionale** (funzioni pure, no classi complesse)
- **LLM-as-judge** tramite API OpenAI per valutazioni qualitative
- **Embeddings OpenAI** (text-embedding-ada-002 o text-embedding-3-large)
- **Gestione test standalone** con salvataggio e confronto risultati

### Caratteristiche Principali

- ✅ **Retrieval**: Precision@k, Recall@k, LLM-as-judge Chunk Relevance
- ✅ **Generation**: Faithfulness (LLM), Answer Relevancy (LLM), Semantic Similarity (embeddings)
- ✅ **Test Standalone**: Ogni configurazione è un test separato e confrontabile
- ✅ **CPU-friendly**: Nessuna GPU necessaria (usa API OpenAI)
- ✅ **Gestione dipendenze**: uv sync per installazione

## 🎯 Domande di Ricerca

La pipeline permette di rispondere sistematicamente a:

1. **Quale LLM genera risposte migliori?** (gpt-4o vs gpt-4o-mini)
   - ✅ NON richiede rebuild del vector store
   
2. **Quale embedding model recupera chunk più rilevanti?** (ada-002 vs 3-large)
   - ⚠️ RICHIEDE rebuild del vector store
   
3. **Quale chunk size è ottimale?** (256, 512, 1024, 2048 token)
   - ⚠️ RICHIEDE rebuild del vector store
   
4. **Quale top_k bilancia qualità e velocità?** (3, 5, 10, 20 chunk)
   - ✅ NON richiede rebuild del vector store

## 📊 Metriche Implementate

### Retrieval (3 metriche)

#### 1. Precision@k
**Definizione**: Proporzione di chunk rilevanti nei top-k risultati.

**Formula**: P@k = (# chunk rilevanti nei top-k) / k

**Interpretazione**:
- **Alto (> 0.7)**: Sistema restituisce prevalentemente risultati rilevanti
- **Medio (0.4-0.7)**: Performance accettabile
- **Basso (< 0.4)**: Troppi falsi positivi

**Livello di certezza**: **CERTO** - Metrica standard nella ricerca IR

#### 2. Recall@k
**Definizione**: Proporzione di chunk rilevanti effettivamente recuperati.

**Formula**: R@k = (# chunk rilevanti nei top-k) / (Totale chunk rilevanti)

**Interpretazione**:
- **Alto (> 0.8)**: Sistema trova la maggior parte dei chunk rilevanti
- **Medio (0.5-0.8)**: Alcuni chunk rilevanti vengono persi
- **Basso (< 0.5)**: Molti chunk rilevanti non recuperati

**Livello di certezza**: **CERTO** - Metrica standard nella ricerca IR

#### 3. LLM-as-Judge Chunk Relevance
**Definizione**: Score medio di rilevanza dei chunk valutati da LLM.

**Implementazione**: Per ogni chunk recuperato, l'LLM valuta la rilevanza rispetto alla query su scala 0-1.

**Interpretazione**:
- **Alto (> 0.7)**: Chunk recuperati sono altamente pertinenti
- **Medio (0.5-0.7)**: Chunk parzialmente pertinenti
- **Basso (< 0.5)**: Chunk poco rilevanti

**Livello di certezza**: **PROBABILE** - Approccio emergente, dipende dalla qualità del prompt

### Generation (3 metriche)

#### 1. Faithfulness (Fedeltà)
**Definizione**: Quanto la risposta è fedele al contesto fornito, senza allucinazioni.

**Implementazione**: LLM-as-judge verifica se tutte le affermazioni nella risposta sono supportate dal contesto.

**Interpretazione**:
- **Alto (> 0.7)**: Risposta ben supportata dal contesto
- **Medio (0.5-0.7)**: Risposta parzialmente supportata
- **Basso (< 0.5)**: Probabile presenza di allucinazioni

**Livello di certezza**: **PROBABILE** - LLM-as-judge è efficace ma non perfetto

#### 2. Answer Relevancy (Rilevanza)
**Definizione**: Quanto la risposta è pertinente alla query dell'utente.

**Implementazione**: LLM-as-judge valuta se la risposta risponde direttamente alla query.

**Interpretazione**:
- **Alto (> 0.7)**: Risposta direttamente pertinente
- **Medio (0.5-0.7)**: Risposta correlata ma incompleta
- **Basso (< 0.5)**: Risposta non allineata con la query

**Livello di certezza**: **PROBABILE** - LLM-as-judge è efficace per questo tipo di valutazione

#### 3. Semantic Similarity (Similarità Semantica)
**Definizione**: Similarità semantica con la risposta di riferimento (ground truth).

**Implementazione**: Calcolo di similarità coseno tra embeddings della risposta generata e della reference answer.

**Interpretazione**:
- **Alto (> 0.8)**: Risposta molto simile alla reference
- **Medio (0.6-0.8)**: Risposta cattura i concetti principali
- **Basso (< 0.6)**: Risposta diverge dalla reference

**Livello di certezza**: **CERTO** - Metrica ben validata, richiede dataset GOLD con reference answers

## 🚀 Setup e Installazione

### 1. Dipendenze

Aggiungi al tuo `pyproject.toml`:

```toml
[tool.uv]
dev-dependencies = [
    "openai>=1.0.0",
    "numpy>=1.24.0",
]
```

Poi esegui:

```bash
uv sync
```

### 2. Configurazione API OpenAI

Imposta la variabile d'ambiente:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

O su Windows:

```cmd
set OPENAI_API_KEY=your-api-key-here
```

### 3. Preparazione Dataset GOLD

Crea `gold_dataset.json`:

```json
[
  {
    "query_id": "q001",
    "query_text": "Qual è la policy di rimborso?",
    "relevant_chunk_ids": ["chunk_001", "chunk_045"],
    "reference_answer": "La policy prevede rimborso completo entro 30 giorni."
  }
]
```

**Requisiti**:
- Minimo 30-50 query per significatività statistica
- `relevant_chunk_ids` devono corrispondere agli ID nel tuo sistema RAG
- `reference_answer` opzionale ma consigliato per Semantic Similarity

## 🔧 Integrazione con Sistema RAG

### Step 1: Personalizza le Funzioni di Retrieval e Generation

Modifica `integration_guide_v2.py`:

```python
def run_retrieval_on_queries(queries, your_rag_system):
    """Esegue retrieval per tutte le query."""
    retrieval_results = []
    
    for query in queries:
        start_time = time.time()
        
        # ⚠️ SOSTITUISCI con la tua implementazione
        retrieved_docs = your_rag_system.retrieve(
            query=query.query_text,
            top_k=10
        )
        
        retrieved_chunk_ids = [doc.chunk_id for doc in retrieved_docs]
        retrieved_chunk_texts = [doc.text for doc in retrieved_docs]
        
        retrieval_time = time.time() - start_time
        
        result = RetrievalResult(
            query_id=query.query_id,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_chunk_texts=retrieved_chunk_texts,
            retrieval_time=retrieval_time
        )
        retrieval_results.append(result)
    
    return retrieval_results


def run_generation_on_queries(queries, retrieval_results, your_rag_system):
    """Esegue generation per tutte le query."""
    generation_results = []
    
    for query in queries:
        # Ottieni chunk dal retrieval
        retrieval_result = next(r for r in retrieval_results if r.query_id == query.query_id)
        
        start_time = time.time()
        
        # ⚠️ SOSTITUISCI con la tua implementazione
        answer = your_rag_system.generate(
            query=query.query_text,
            context_chunks=retrieval_result.retrieved_chunk_texts
        )
        
        generation_time = time.time() - start_time
        
        result = GenerationResult(
            query_id=query.query_id,
            generated_answer=answer,
            context_chunks=retrieval_result.retrieved_chunk_texts,
            generation_time=generation_time
        )
        generation_results.append(result)
    
    return generation_results
```

### Step 2: Esegui un Test

```python
from integration_guide_v2 import run_single_test

# Inizializza il tuo sistema RAG
your_rag = YourRAGSystem(
    embedding_model="text-embedding-3-large",
    llm_model="gpt-4o-mini",
    chunk_size=512,
    top_k=5
)

# Definisci configurazione
configuration = {
    "name": "Test Baseline",
    "embedding_model": "text-embedding-3-large",
    "llm_model": "gpt-4o-mini",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k": 5
}

# Esegui test completo
test_result = run_single_test(
    gold_dataset_path="gold_dataset.json",
    your_rag_system=your_rag,
    configuration=configuration
)
```

## 📈 Workflow per Esperimenti di Tesi

### Fase 1: Baseline (Test 1)

Stabilisci una configurazione di riferimento:

```python
config_baseline = {
    "name": "Baseline",
    "embedding_model": "text-embedding-3-large",
    "llm_model": "gpt-4o-mini",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k": 5
}
```

### Fase 2: Variazioni che NON Richiedono Rebuild

Test che puoi eseguire senza ricostruire il vector store:

**Test 2 - LLM più potente:**
```python
config_gpt4o = {
    "name": "LLM Upgrade - gpt-4o",
    "embedding_model": "text-embedding-3-large",
    "llm_model": "gpt-4o",  # ← Solo questo cambia
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k": 5
}
```

**Test 3-4 - Variazione top_k:**
```python
config_k3 = {..., "top_k": 3, "name": "Top-K = 3"}
config_k10 = {..., "top_k": 10, "name": "Top-K = 10"}
```

### Fase 3: Variazioni che RICHIEDONO Rebuild

⚠️ **IMPORTANTE**: Prima di ogni test, devi ricostruire il vector store manualmente.

**Test 5 - Embedding diverso:**
```python
# 1. Ricostruisci vector store con text-embedding-ada-002
# 2. Poi esegui test:
config_ada = {
    "name": "Embedding - ada-002",
    "embedding_model": "text-embedding-ada-002",
    "llm_model": "gpt-4o-mini",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k": 5
}
```

**Test 6-8 - Chunk size diverso:**
```python
# Per ogni test: riprocessa documenti e ricostruisci vector store

config_chunk256 = {..., "chunk_size": 256, "chunk_overlap": 25}
config_chunk1024 = {..., "chunk_size": 1024, "chunk_overlap": 100}
config_chunk2048 = {..., "chunk_size": 2048, "chunk_overlap": 200}
```

### Fase 4: Confronto Risultati

```bash
python compare_results.py
```

O in codice:

```python
from integration_guide_v2 import confronta_tutti_i_risultati

confronta_tutti_i_risultati()
```

## 📊 Analisi e Interpretazione Risultati

### Tabella Comparativa

Il sistema genera automaticamente una tabella come questa:

```
Test / Config                          P@5      R@5   LLM-Rel    Faith    Relev   SemSim
----------------------------------------------------------------------------------------
Baseline                            0.7500   0.6000   0.8200   0.8500   0.7800   0.7200
LLM Upgrade - gpt-4o                0.7500   0.6000   0.8200   0.9100   0.8500   0.7800
Embedding - ada-002                 0.6800   0.5400   0.7500   0.8400   0.7700   0.7100
```

### Interpretazione

**Se Precision alta ma Recall basso:**
- Il sistema è conservativo, restituisce solo risultati molto rilevanti
- Rischio di perdere informazioni importanti
- **Azione**: Aumenta top_k o modifica strategia di retrieval

**Se Faithfulness basso:**
- Il modello sta "allucinando" o aggiungendo informazioni
- **Azione**: Usa LLM più potente o migliora prompt di generation

**Se Answer Relevancy basso:**
- Le risposte non sono allineate con le query
- **Azione**: Migliora prompt o usa contesto più pertinente

### Trade-off Identificati

1. **Precision vs Recall**: Spesso inversamente correlati
2. **Qualità vs Costo**: gpt-4o migliore ma più costoso di gpt-4o-mini
3. **Granularità Chunk**: Chunk grandi = più contesto ma meno precisione

## 🛠️ Comandi Utili

```bash
# Confronta tutti i risultati
python compare_results.py

# Analizza per parametro specifico
python compare_results.py --analyze-param llm_model
python compare_results.py --analyze-param chunk_size

# Esporta in CSV per Excel
python compare_results.py --export-csv results.csv

# Mostra dettagli di tutti i test
python compare_results.py --show-details
```

## 📁 Struttura File

```
rag_evaluation_pipeline_v2.py   # Pipeline principale con metriche
integration_guide_v2.py          # Guida e template integrazione
compare_results.py               # Script per confronto risultati
README.md                        # Questa documentazione
gold_dataset_template.json       # Template dataset GOLD
evaluation_results/              # Directory risultati salvati (auto-creata)
  ├── test_20260119_143022.json
  ├── test_20260119_150315.json
  └── ...
```

## ❓ FAQ

**Q: Quanto costa eseguire un test?**  
A: Dipende dal numero di query e dal modello LLM. Con gpt-4o-mini e 50 query: ~$0.50-1.00

**Q: Posso usare un altro LLM per LLM-as-judge?**  
A: Sì, modifica il parametro `llm_model` in `run_full_evaluation()`. Supporta tutti i modelli OpenAI.

**Q: Come gestisco test con chunking diverso?**  
A: Devi ricostruire il vector store manualmente prima del test. Ogni test è standalone.

**Q: I risultati sono riproducibili?**  
A: Sì, usando `temperature=0.0` per le chiamate LLM. Lievi variazioni possono verificarsi per caching API.

**Q: Quanto tempo richiede un test completo?**  
A: Con 50 query: 5-10 minuti (dipende da latenza API e numero chunk da valutare).

## 📚 Contributo alla Tesi

### Decisioni di Design Giustificabili

1. **LLM-as-judge vs Rule-based**: 
   - **PROBABILE** che catturi meglio sfumature semantiche
   - Trade-off: più flessibile ma meno deterministico

2. **Embeddings vs Manual Scoring**:
   - **CERTO** che sia più scalabile
   - Limitazione: può non catturare alcune sfumature

3. **Test Standalone vs Pipeline Continua**:
   - **CERTO** che sia più flessibile per ricerca
   - Permette confronti puliti tra configurazioni diverse

### Limitazioni da Documentare

1. LLM-as-judge può avere bias propri
2. Semantic Similarity richiede reference answers di qualità
3. Risultati dipendono dalla qualità del dataset GOLD
4. Costi API possono crescere con dataset grandi

## 📧 Supporto

Per problemi di integrazione, consulta:
1. `integration_guide_v2.py` per esempi dettagliati
2. Commenti inline nel codice
3. Workflow completo nella guida

---

**Versione**: 2.0.0  
**Data**: Gennaio 2026  
**Licenza**: Academic Use  
**Autore**: [Nome Studente]
