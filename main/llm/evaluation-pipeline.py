from typing import List, Set, Sequence, Tuple, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer, util as st_util
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

"""
File: evaluation-pipeline.py

Evaluation pipeline for a RAG system:
- Retrieval metrics: context precision@k, context recall@k
- Evaluation metrics:
    - Faithfulness (NLI entailment of answer given retrieved contexts)
    - Relevance (semantic similarity answer <-> contexts)
    - Answer semantic similarity (answer <-> reference answer(s))

Dependencies:
    pip install sentence-transformers transformers torch numpy scipy

Usage: import functions from this file or run the demo in __main__.
"""


# Embedding & similarity
try:
except Exception as e:
    raise ImportError("Install sentence-transformers: pip install sentence-transformers") from e

# NLI model
try:
    import torch.nn.functional as F
except Exception as e:
    raise ImportError("Install transformers and torch: pip install transformers torch") from e


# ------------------
# Retrieval metrics
# ------------------

def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """
    Compute context precision@k: fraction of top-k retrieved that are relevant.
    """
    if k <= 0:
        return 0.0
    topk = list(retrieved_ids)[:k]
    if len(topk) == 0:
        return 0.0
    hits = sum(1 for doc in topk if doc in relevant_ids)
    return hits / len(topk)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """
    Compute context recall@k: fraction of relevant items retrieved in top-k.
    If no relevant items exist, returns 0.0.
    """
    if len(relevant_ids) == 0:
        return 0.0
    topk = set(list(retrieved_ids)[:k])
    hits = sum(1 for doc in relevant_ids if doc in topk)
    return hits / len(relevant_ids)


# ------------------
# Semantic similarity (embeddings)
# ------------------

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """
        Returns L2-normalized embeddings as numpy array (n x dim).
        """
        embs = self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return embs

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Cosine similarity between arrays of normalized embeddings.
        If a shape is (m, d) and b shape is (n, d), returns (m, n).
        """
        return np.dot(a, b.T)


# ------------------
# NLI faithfulness
# ------------------

class NLIModel:
    """
    NLI model wrapper using a sequence classification model trained on MNLI.
    Computes entailment probability P(entailment | premise=context, hypothesis=answer).
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        # MNLI label mapping in HuggingFace models is typically: 0: contradiction, 1: neutral, 2: entailment
        self.entailment_label_index = 2

    def entailment_prob(self, premise: str, hypothesis: str) -> float:
        """
        Returns probability of entailment (float in [0,1]) for a single pair.
        """
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, num_labels)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        return float(probs[self.entailment_label_index])


# ------------------
# Evaluation metrics
# ------------------

def answer_semantic_similarity(answer: str, references: Sequence[str], embedder: EmbeddingModel) -> float:
    """
    Compute semantic similarity between answer and reference answers.
    Returns the maximum cosine similarity across references (in [0,1]).
    """
    if len(references) == 0:
        return 0.0
    embs = embedder.embed([answer] + list(references))
    a = embs[0:1]  # (1,d)
    refs = embs[1:]  # (n,d)
    sims = EmbeddingModel.cosine_sim(a, refs)[0]
    return float(np.max(sims))


def relevance_score(answer: str, contexts: Sequence[str], embedder: EmbeddingModel, agg: str = "max") -> float:
    """
    Relevance of answer to retrieved contexts using embedding similarity.
    - agg: "max" or "mean"
    Returns score in [0,1].
    """
    if len(contexts) == 0:
        return 0.0
    embs = embedder.embed([answer] + list(contexts))
    a = embs[0:1]
    ctxs = embs[1:]
    sims = EmbeddingModel.cosine_sim(a, ctxs)[0]
    if agg == "mean":
        return float(np.mean(sims))
    return float(np.max(sims))


def faithfulness_metrics(answer: str, contexts: Sequence[str], nli_model: NLIModel, agg: str = "max") -> float:
    """
    Compute faithfulness score as aggregated entailment probability across contexts.
    - For each context (premise) we compute P(entailment | premise, hypothesis=answer).
    - agg: "max" or "mean" to aggregate multiple contexts into a single score.
    """
    if len(contexts) == 0:
        return 0.0
    probs = [nli_model.entailment_prob(context, answer) for context in contexts]
    if agg == "mean":
        return float(np.mean(probs))
    return float(np.max(probs))


# ------------------
# Combined evaluation for a single QA instance
# ------------------

def evaluate_instance(
    retrieved_doc_ids: Sequence[str],
    retrieved_texts: Sequence[str],
    relevant_doc_ids: Set[str],
    answer: str,
    reference_answers: Sequence[str],
    embedder: EmbeddingModel,
    nli_model: NLIModel,
    k_values: Sequence[int] = (1, 3, 5)
) -> Dict:
    """
    Evaluate one query/answer:
    - retrieval metrics: precision@k and recall@k for provided k_values
    - relevance, faithfulness, answer similarity (aggregated)
    Returns dictionary with all scores.
    """
    out = {"retrieval": {}, "evaluation": {}}
    for k in k_values:
        out["retrieval"][f"precision@{k}"] = precision_at_k(retrieved_doc_ids, relevant_doc_ids, k)
        out["retrieval"][f"recall@{k}"] = recall_at_k(retrieved_doc_ids, relevant_doc_ids, k)

    # Use top-K contexts for evaluation; choose k = max(k_values) or all provided contexts if fewer
    use_k = min(len(retrieved_texts), max(k_values))
    contexts_topk = list(retrieved_texts)[:use_k]

    out["evaluation"]["answer_semantic_similarity"] = answer_semantic_similarity(answer, reference_answers, embedder)
    out["evaluation"]["relevance_max"] = relevance_score(answer, contexts_topk, embedder, agg="max")
    out["evaluation"]["relevance_mean"] = relevance_score(answer, contexts_topk, embedder, agg="mean")
    out["evaluation"]["faithfulness_max"] = faithfulness_metrics(answer, contexts_topk, nli_model, agg="max")
    out["evaluation"]["faithfulness_mean"] = faithfulness_metrics(answer, contexts_topk, nli_model, agg="mean")

    return out


# ------------------
# Demo / example
# ------------------

if __name__ == "__main__":
    # Example usage with toy data
    embedder = EmbeddingModel("all-MiniLM-L6-v2")
    nli = NLIModel("facebook/bart-large-mnli")

    retrieved_ids = ["doc1", "doc2", "doc3", "doc4"]
    retrieved_texts = [
        "Paris is the capital of France. It is known for the Eiffel Tower.",
        "Berlin is the capital of Germany and has the Brandenburg Gate.",
        "The capital of Italy is Rome, famous for the Colosseum.",
        "Madrid is the capital of Spain and famous for its museums.",
    ]
    relevant_ids = {"doc1", "doc3"}  # ground-truth relevant contexts
    answer = "Paris is the capital city of France and home to the Eiffel Tower."
    references = ["Paris is France's capital.", "The Eiffel Tower is in Paris."]

    res = evaluate_instance(
        retrieved_doc_ids=retrieved_ids,
        retrieved_texts=retrieved_texts,
        relevant_doc_ids=relevant_ids,
        answer=answer,
        reference_answers=references,
        embedder=embedder,
        nli_model=nli,
        k_values=(1, 2, 3),
    )

    print(json.dumps(res, indent=2))