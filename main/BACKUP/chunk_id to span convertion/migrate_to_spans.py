"""
migrate_to_spans.py
===================
Migra il gold dataset da chunk-ID based a span-text based.

Per ogni query del gold dataset, recupera dal DB il testo dei chunk
identificati da relevant_chunk_ids e li salva come relevant_spans.
I vecchi campi vengono mantenuti per compatibilità.

Posizionamento: metti questo file nella stessa cartella di evaluation_pipeline.py
(es. main/evaluation/). Lo script importa get_connection() da file_embedding/db_connection.py
esattamente come fanno search.py e llm.py.

Uso:
    python migrate_to_spans.py
    python migrate_to_spans.py --input gold_dataset_v2.json --output gold_dataset_v3.json

Output: gold_dataset_v3.json con il nuovo campo relevant_spans aggiunto.

Dopo la migrazione, usa span_refinement_prompt.txt per snellire gli span
con Claude in una chat separata.
"""

import json
import argparse
import os
import sys
from pathlib import Path

# Stesso pattern di import usato in search.py e llm.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from file_embedding.db_connection import get_connection


# ── Recupero testo chunk ──────────────────────────────────────────────────────

def fetch_chunk_text(cursor, num_ri: str, progressivo: int):
    """
    Recupera il testo di un chunk dato NumRI e Progressivo.
    Ritorna None se il chunk non esiste nel DB.
    """
    cursor.execute(
        "SELECT Content FROM DocumentChunks WHERE NumRI = ? AND Progressivo = ?",
        (int(num_ri), int(progressivo))
    )
    row = cursor.fetchone()
    return row[0] if row else None


# ── Migrazione ────────────────────────────────────────────────────────────────

def migrate_dataset(data: list, cursor, verbose: bool = True) -> list:
    missing_total = 0

    for q in data:
        chunk_ids = q.get("relevant_chunk_ids", [])

        if not chunk_ids:
            q["relevant_spans"] = []
            continue

        spans = []
        missing_for_query = []

        for chunk_id in chunk_ids:
            try:
                num_ri, progressivo = chunk_id.split("_")
                text = fetch_chunk_text(cursor, num_ri, int(progressivo))
                if text:
                    spans.append(text)
                else:
                    missing_for_query.append(chunk_id)
            except ValueError:
                print(f"  Formato chunk_id non valido: {chunk_id!r} -- skip")

        q["relevant_spans"] = spans

        if missing_for_query:
            missing_total += len(missing_for_query)
            print(f"  [{q['query_id']}] Chunk non trovati nel DB: {missing_for_query}")
            print(f"    Il DB potrebbe essere stato re-indicizzato con chunking diverso.")
            print(f"    Span recuperati: {len(spans)}/{len(chunk_ids)}")

        if verbose and spans:
            avg_len = sum(len(s) for s in spans) // len(spans)
            print(f"  [{q['query_id']}] {len(spans)} span recuperati "
                  f"(lunghezza media: {avg_len} char)")

    if missing_total > 0:
        print(f"\nTotale chunk non trovati: {missing_total}")

    return data


def print_summary(data: list):
    positive           = [q for q in data if not q.get("expected_behavior")]
    negative           = [q for q in data if     q.get("expected_behavior")]
    total_spans        = sum(len(q.get("relevant_spans", [])) for q in positive)
    queries_with_spans = sum(1 for q in positive if q.get("relevant_spans"))

    print(f"\n{'-'*55}")
    print(f"  Query totali         : {len(data)}")
    print(f"  Query positive       : {len(positive)}")
    print(f"  Query negative       : {len(negative)}")
    print(f"  Query con span       : {queries_with_spans}/{len(positive)}")
    print(f"  Span totali          : {total_spans}")
    if positive:
        print(f"  Span medi per query  : {total_spans / len(positive):.1f}")
    print(f"{'-'*55}")

    no_spans = [q for q in positive if not q.get("relevant_spans")]
    if no_spans:
        print(f"\n  Query positive senza span ({len(no_spans)}):")
        for q in no_spans:
            print(f"    - {q['query_id']}: {q['query_text'][:60]}...")

    print(f"\nProssimo step: segui le istruzioni in span_refinement_prompt.txt")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Migra gold dataset da chunk ID a span testuali"
    )
    parser.add_argument("--input",  default="gold_dataset_v2.json",
                        help="File di input (default: gold_dataset_v2.json)")
    parser.add_argument("--output", default="gold_dataset_v3.json",
                        help="File di output (default: gold_dataset_v3.json)")
    parser.add_argument("--quiet",  action="store_true",
                        help="Output meno verboso")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Errore: file non trovato: {input_path}")
        return

    print(f"Caricamento: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Query caricate: {len(data)}\n")

    print("Connessione al DB...")
    try:
        conn   = get_connection()
        cursor = conn.cursor()
    except Exception as e:
        print(f"Errore connessione DB: {e}")
        return

    print("Recupero span dal DB...\n")
    data = migrate_dataset(data, cursor, verbose=not args.quiet)

    cursor.close()
    conn.close()

    print_summary(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSalvato: {output_path}")


if __name__ == "__main__":
    main()