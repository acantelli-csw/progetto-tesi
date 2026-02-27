import json

with open("gold_dataset_v3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("span_refinement_output.json", "r", encoding="utf-8") as f:
    refined = json.load(f)

# Costruisce un indice per accesso rapido
refined_index = {
    (r["query_id"], r["span_index"]): r["refined_span"]
    for r in refined
}

for q in data:
    spans = q.get("relevant_spans", [])
    if not spans:
        continue
    new_spans = []
    for i, span in enumerate(spans):
        key = (q["query_id"], i)
        refined_text = refined_index.get(key, span)  # fallback allo span originale
        if refined_text:  # scarta span vuoti (LLM ha giudicato non pertinente)
            new_spans.append(refined_text)
    q["relevant_spans"]          = new_spans
    q["relevant_spans_original"] = spans  # backup per confronto

with open("gold_dataset_v4.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Salvato gold_dataset_v4.json")
removed = sum(
    len(q.get("relevant_spans_original", [])) - len(q.get("relevant_spans", []))
    for q in data
)
print(f"Span rimossi perché non pertinenti: {removed}")