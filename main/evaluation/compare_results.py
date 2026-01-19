"""
Script per Confronto Risultati Test
====================================

Script standalone per caricare e confrontare tutti i risultati
di test salvati, generando analisi comparative.

Uso:
    python compare_results.py
    python compare_results.py --results-dir custom_results/
"""

import argparse
from pathlib import Path
from main.llm.rag_evaluation_pipeline_v2 import (
    load_all_test_results,
    compare_test_results,
    print_test_result
)


def analizza_trend_metriche(results):
    """
    Analizza i trend delle metriche nel tempo.
    
    Args:
        results: Lista di TestResult ordinati per timestamp
    """
    if len(results) < 2:
        print("\n⚠️  Servono almeno 2 test per analizzare trend")
        return
    
    print("\n" + "="*70)
    print("ANALISI TREND METRICHE")
    print("="*70)
    
    # Estrai metriche chiave
    precision_trend = [r.retrieval_metrics.get('avg_precision_at_5', 0) for r in results]
    faithfulness_trend = [r.generation_metrics.get('avg_faithfulness', 0) for r in results]
    
    # Calcola variazioni
    p_change = ((precision_trend[-1] - precision_trend[0]) / precision_trend[0] * 100) if precision_trend[0] > 0 else 0
    f_change = ((faithfulness_trend[-1] - faithfulness_trend[0]) / faithfulness_trend[0] * 100) if faithfulness_trend[0] > 0 else 0
    
    print(f"\nDal primo all'ultimo test:")
    print(f"  • Precision@5: {precision_trend[0]:.4f} → {precision_trend[-1]:.4f} ({p_change:+.1f}%)")
    print(f"  • Faithfulness: {faithfulness_trend[0]:.4f} → {faithfulness_trend[-1]:.4f} ({f_change:+.1f}%)")
    
    # Identifica miglior e peggior test
    best_p_idx = precision_trend.index(max(precision_trend))
    worst_p_idx = precision_trend.index(min(precision_trend))
    
    print(f"\n  🏆 Miglior Precision@5: Test {best_p_idx + 1} ({results[best_p_idx].configuration.get('name', 'N/A')})")
    print(f"  📉 Peggior Precision@5: Test {worst_p_idx + 1} ({results[worst_p_idx].configuration.get('name', 'N/A')})")


def confronta_configurazioni_per_parametro(results, parameter_name):
    """
    Confronta risultati raggruppati per un parametro specifico.
    
    Args:
        results: Lista di TestResult
        parameter_name: Nome del parametro da analizzare (es: 'llm_model', 'chunk_size')
    """
    print(f"\n" + "="*70)
    print(f"ANALISI PER PARAMETRO: {parameter_name}")
    print("="*70)
    
    # Raggruppa per valore del parametro
    groups = {}
    for result in results:
        param_value = result.configuration.get(parameter_name, 'N/A')
        if param_value not in groups:
            groups[param_value] = []
        groups[param_value].append(result)
    
    if len(groups) <= 1:
        print(f"\n⚠️  Un solo valore per '{parameter_name}', confronto non possibile")
        return
    
    # Calcola medie per gruppo
    print(f"\nConfronto per {parameter_name}:")
    print("-"*70)
    
    for param_value, group_results in groups.items():
        avg_precision = sum(r.retrieval_metrics.get('avg_precision_at_5', 0) for r in group_results) / len(group_results)
        avg_faithfulness = sum(r.generation_metrics.get('avg_faithfulness', 0) for r in group_results) / len(group_results)
        
        print(f"\n  {parameter_name} = {param_value} ({len(group_results)} test):")
        print(f"    • Avg Precision@5: {avg_precision:.4f}")
        print(f"    • Avg Faithfulness: {avg_faithfulness:.4f}")


def genera_raccomandazioni(results):
    """
    Genera raccomandazioni basate sui risultati.
    
    Args:
        results: Lista di TestResult
    """
    if not results:
        return
    
    print("\n" + "="*70)
    print("RACCOMANDAZIONI")
    print("="*70)
    
    # Trova configurazione con migliore bilanciamento
    def score_bilanciamento(result):
        # Score basato su media di metriche chiave normalizzate
        p5 = result.retrieval_metrics.get('avg_precision_at_5', 0)
        r5 = result.retrieval_metrics.get('avg_recall_at_5', 0)
        faith = result.generation_metrics.get('avg_faithfulness', 0)
        relev = result.generation_metrics.get('avg_answer_relevancy', 0)
        return (p5 + r5 + faith + relev) / 4
    
    best_balanced = max(results, key=score_bilanciamento)
    
    print("\n🎯 CONFIGURAZIONE RACCOMANDATA (miglior bilanciamento):")
    print(f"   Nome: {best_balanced.configuration.get('name', 'N/A')}")
    print(f"   Test ID: {best_balanced.test_id}")
    print("\n   Parametri:")
    for key, value in best_balanced.configuration.items():
        if key != 'name' and key != 'note':
            print(f"     • {key}: {value}")
    
    print(f"\n   Performance:")
    print(f"     • Precision@5: {best_balanced.retrieval_metrics.get('avg_precision_at_5', 0):.4f}")
    print(f"     • Recall@5: {best_balanced.retrieval_metrics.get('avg_recall_at_5', 0):.4f}")
    print(f"     • Faithfulness: {best_balanced.generation_metrics.get('avg_faithfulness', 0):.4f}")
    print(f"     • Answer Relevancy: {best_balanced.generation_metrics.get('avg_answer_relevancy', 0):.4f}")
    
    # Trade-off analysis
    print("\n⚖️  TRADE-OFF IDENTIFICATI:")
    
    best_precision = max(results, key=lambda r: r.retrieval_metrics.get('avg_precision_at_5', 0))
    best_recall = max(results, key=lambda r: r.retrieval_metrics.get('avg_recall_at_5', 0))
    
    if best_precision.test_id != best_recall.test_id:
        print(f"\n   • Precision vs Recall:")
        print(f"     - Massima precision: {best_precision.configuration.get('name', 'N/A')}")
        print(f"       (P@5={best_precision.retrieval_metrics.get('avg_precision_at_5', 0):.4f}, "
              f"R@5={best_precision.retrieval_metrics.get('avg_recall_at_5', 0):.4f})")
        print(f"     - Massimo recall: {best_recall.configuration.get('name', 'N/A')}")
        print(f"       (P@5={best_recall.retrieval_metrics.get('avg_precision_at_5', 0):.4f}, "
              f"R@5={best_recall.retrieval_metrics.get('avg_recall_at_5', 0):.4f})")
    
    # Analisi costi (se disponibile info su modelli)
    llm_models = [r.configuration.get('llm_model') for r in results]
    if 'gpt-4o' in llm_models and 'gpt-4o-mini' in llm_models:
        print(f"\n   • Costo vs Qualità:")
        print(f"     Considera il trade-off tra qualità (gpt-4o) e costo (gpt-4o-mini)")


def esporta_per_excel(results, output_file="comparison_export.csv"):
    """
    Esporta i risultati in formato CSV per analisi in Excel.
    
    Args:
        results: Lista di TestResult
        output_file: Nome del file CSV di output
    """
    if not results:
        return
    
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Test ID',
            'Timestamp',
            'Config Name',
            'Embedding Model',
            'LLM Model',
            'Chunk Size',
            'Top K',
            'Precision@1',
            'Precision@3',
            'Precision@5',
            'Precision@10',
            'Recall@1',
            'Recall@3',
            'Recall@5',
            'Recall@10',
            'LLM Chunk Relevance',
            'Faithfulness',
            'Answer Relevancy',
            'Semantic Similarity',
            'Avg Retrieval Time',
            'Avg Generation Time',
            'Total Eval Time'
        ])
        
        # Righe
        for r in results:
            writer.writerow([
                r.test_id,
                r.timestamp,
                r.configuration.get('name', ''),
                r.configuration.get('embedding_model', ''),
                r.configuration.get('llm_model', ''),
                r.configuration.get('chunk_size', ''),
                r.configuration.get('top_k', ''),
                r.retrieval_metrics.get('avg_precision_at_1', ''),
                r.retrieval_metrics.get('avg_precision_at_3', ''),
                r.retrieval_metrics.get('avg_precision_at_5', ''),
                r.retrieval_metrics.get('avg_precision_at_10', ''),
                r.retrieval_metrics.get('avg_recall_at_1', ''),
                r.retrieval_metrics.get('avg_recall_at_3', ''),
                r.retrieval_metrics.get('avg_recall_at_5', ''),
                r.retrieval_metrics.get('avg_recall_at_10', ''),
                r.retrieval_metrics.get('avg_avg_chunk_relevance_top5', ''),
                r.generation_metrics.get('avg_faithfulness', ''),
                r.generation_metrics.get('avg_answer_relevancy', ''),
                r.generation_metrics.get('avg_semantic_similarity', ''),
                r.retrieval_metrics.get('avg_retrieval_time', ''),
                r.generation_metrics.get('avg_generation_time', ''),
                r.total_evaluation_time
            ])
    
    print(f"\n✓ Risultati esportati in: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Confronta risultati di test salvati"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='evaluation_results',
        help='Directory contenente i risultati (default: evaluation_results)'
    )
    parser.add_argument(
        '--export-csv',
        type=str,
        help='Esporta risultati in CSV per Excel'
    )
    parser.add_argument(
        '--analyze-param',
        type=str,
        help='Analizza risultati per parametro specifico (es: llm_model, chunk_size)'
    )
    parser.add_argument(
        '--show-details',
        action='store_true',
        help='Mostra dettagli completi di ogni test'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("CONFRONTO RISULTATI TEST - Sistema RAG")
    print("="*70)
    
    # Carica risultati
    print(f"\nCaricamento risultati da: {args.results_dir}/")
    results = load_all_test_results(args.results_dir)
    
    if not results:
        print(f"\n⚠️  Nessun risultato trovato in {args.results_dir}/")
        print("\nAssicurati di:")
        print("1. Aver eseguito almeno un test con run_single_test()")
        print("2. Specificare la directory corretta con --results-dir")
        return
    
    print(f"✓ Caricati {len(results)} test")
    
    # Ordina per timestamp
    results.sort(key=lambda r: r.timestamp)
    
    # Mostra dettagli se richiesto
    if args.show_details:
        print("\n" + "="*70)
        print("DETTAGLI TEST")
        print("="*70)
        for i, result in enumerate(results, 1):
            print(f"\n--- Test {i}/{len(results)} ---")
            print_test_result(result)
    
    # Tabella comparativa
    compare_test_results(results)
    
    # Analisi trend
    analizza_trend_metriche(results)
    
    # Analisi per parametro specifico
    if args.analyze_param:
        confronta_configurazioni_per_parametro(results, args.analyze_param)
    
    # Raccomandazioni
    genera_raccomandazioni(results)
    
    # Export CSV
    if args.export_csv:
        esporta_per_excel(results, args.export_csv)
    
    print("\n" + "="*70)
    print("ANALISI COMPLETATA")
    print("="*70)
    print("\nComandi utili:")
    print("  • Analizza per LLM:        python compare_results.py --analyze-param llm_model")
    print("  • Analizza per chunk size: python compare_results.py --analyze-param chunk_size")
    print("  • Esporta in Excel:        python compare_results.py --export-csv results.csv")
    print("  • Mostra tutti i dettagli: python compare_results.py --show-details")


if __name__ == "__main__":
    main()
