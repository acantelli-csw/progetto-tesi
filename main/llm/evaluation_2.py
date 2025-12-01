dataset = []

for q in test_queries:
    answer, chunks = my_custom_rag.query(q)  # la tua funzione RAG
    gt = generate_ground_truth(q)  # opzionale
    dataset.append({
        "question": q,
        "response": answer,
        "contexts": chunks,
        "ground_truth": gt
    })


from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall

result_ragas = evaluate(dataset, metrics=[
    answer_relevancy, faithfulness, context_precision, context_recall
])
print(result_ragas)

from llama_index.evaluation import Evaluation

# Definisci una classe che simula il tuo RAG
class MyCustomRAG:
    def query(self, question):
        answer, context = my_custom_rag.query(question)
        return {"answer": answer, "context": context}

my_rag_eval = MyCustomRAG()

# Crea le valutazioni
eval = Evaluation(
    queries=[d["question"] for d in dataset],
    responses=[my_rag_eval.query(d["question"]) for d in dataset],
    ground_truths=[d["ground_truth"] for d in dataset]
)

metrics = eval.compute_metrics()
print(metrics)
