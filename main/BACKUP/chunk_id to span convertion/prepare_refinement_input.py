import json

with open("gold_dataset_v3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

refinement_input = []
for q in data:
    for i, span in enumerate(q.get("relevant_spans", [])):
        refinement_input.append({
            "query_id":      q["query_id"],
            "query_text":    q["query_text"],
            "span_index":    i,
            "original_span": span
        })

with open("span_refinement_input.json", "w", encoding="utf-8") as f:
    json.dump(refinement_input, f, indent=2, ensure_ascii=False)

print(f"Generati {len(refinement_input)} span da snellire")