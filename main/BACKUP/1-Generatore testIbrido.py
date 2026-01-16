import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

class QuestionTier(Enum):
    """Livelli di qualità/difficoltà del testset"""
    GOLD = "gold_standard"      # Domande con ground truth verificato
    SILVER = "silver_standard"  # Domande generate da docs rilevanti
    STRESS = "stress_test"      # Domande generate da corpus completo

@dataclass
class GroundTruthDocument:
    """Documento di riferimento per un tema"""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float = 1.0  # 0-1, quanto è rilevante per il tema

@dataclass
class ThematicDomain:
    """Dominio tematico con documenti di riferimento"""
    theme_id: str
    theme_name: str
    theme_description: str
    reference_docs: List[GroundTruthDocument]
    
    def get_doc_ids(self) -> List[str]:
        return [doc.doc_id for doc in self.reference_docs]

@dataclass
class TestQuestion:
    """Domanda del testset con metadati estesi"""
    question_id: str
    question: str
    tier: QuestionTier
    theme_id: str = None
    ground_truth_doc_ids: List[str] = None
    reference_answer: str = None
    
    # Campi popolati dal retriever
    retrieved_contexts: List[str] = None
    retrieved_doc_ids: List[str] = None
    retrieved_count: int = 0
    
    # Metriche calcolabili
    metrics: Dict[str, float] = None
    
    def to_dict(self):
        return asdict(self)


class HybridTestsetGenerator:
    """
    Generatore di testset ibrido che combina:
    1. Domande gold con ground truth noto
    2. Domande silver generate da docs rilevanti
    3. Domande stress dal corpus completo
    """
    
    def __init__(self, 
                 thematic_domains: List[ThematicDomain],
                 ragas_generator,  # Il tuo generator Ragas esistente
                 retrieve_fn):
        self.domains = thematic_domains
        self.ragas_generator = ragas_generator
        self.retrieve_fn = retrieve_fn
        
    def generate_gold_questions(self, 
                                n_per_theme: int = 1,
                                manual_mode: bool = False) -> List[TestQuestion]:
        """
        Genera domande GOLD con ground truth verificato.
        
        Args:
            n_per_theme: Numero di domande per tema
            manual_mode: Se True, richiede inserimento manuale (per tesi!)
        """
        gold_questions = []
        
        for domain in self.domains:
            print(f"\n📌 Tema: {domain.theme_name}")
            print(f"   Descrizione: {domain.theme_description}")
            print(f"   Documenti di riferimento: {len(domain.reference_docs)}")
            
            if manual_mode:
                # Modalità manuale per massima qualità (ideale per tesi)
                for i in range(n_per_theme):
                    print(f"\n   Domanda {i+1}/{n_per_theme}:")
                    question_text = input("   → Inserisci domanda: ")
                    reference_answer = input("   → Risposta attesa (opzionale): ")
                    
                    question = TestQuestion(
                        question_id=f"gold_{domain.theme_id}_{i+1}",
                        question=question_text,
                        tier=QuestionTier.GOLD,
                        theme_id=domain.theme_id,
                        ground_truth_doc_ids=domain.get_doc_ids(),
                        reference_answer=reference_answer if reference_answer else None
                    )
                    gold_questions.append(question)
            else:
                # Modalità automatica con Ragas sui docs di riferimento
                from llama_index.core.schema import Document as LlamaDocument
                
                llama_docs = [
                    LlamaDocument(
                        text=doc.content,
                        metadata=doc.metadata,
                        id_=doc.doc_id
                    )
                    for doc in domain.reference_docs
                ]
                
                # Genera con Ragas SOLO da questi documenti
                testset = self.ragas_generator.generate_with_llamaindex_docs(
                    documents=llama_docs,
                    testset_size=n_per_theme,
                    raise_exceptions=False
                )
                
                for i, example in enumerate(testset.to_list()[:n_per_theme]):
                    question = TestQuestion(
                        question_id=f"gold_{domain.theme_id}_{i+1}",
                        question=example.get('user_input', example.get('question')),
                        tier=QuestionTier.GOLD,
                        theme_id=domain.theme_id,
                        ground_truth_doc_ids=domain.get_doc_ids(),
                        reference_answer=example.get('reference')
                    )
                    gold_questions.append(question)
        
        return gold_questions
    
    def generate_silver_questions(self, n_total: int = 10) -> List[TestQuestion]:
        """
        Genera domande SILVER da documenti rilevanti (ma non ground truth assoluto).
        Utile per testare capacità di discriminazione del retriever.
        """
        silver_questions = []
        
        # Strategia: genera da subset espanso (docs rilevanti + alcuni vicini)
        for i, domain in enumerate(self.domains[:n_total]):
            # Prendi docs di riferimento + alcuni random per "noise"
            ref_doc_ids = domain.get_doc_ids()
            
            from llama_index.core.schema import Document as LlamaDocument
            llama_docs = [
                LlamaDocument(
                    text=doc.content,
                    metadata=doc.metadata,
                    id_=doc.doc_id
                )
                for doc in domain.reference_docs
            ]
            
            testset = self.ragas_generator.generate_with_llamaindex_docs(
                documents=llama_docs,
                testset_size=1,
                raise_exceptions=False
            )
            
            if testset:
                example = testset.to_list()[0]
                question = TestQuestion(
                    question_id=f"silver_{i+1}",
                    question=example.get('user_input', example.get('question')),
                    tier=QuestionTier.SILVER,
                    theme_id=domain.theme_id,
                    ground_truth_doc_ids=ref_doc_ids,
                    reference_answer=example.get('reference')
                )
                silver_questions.append(question)
        
        return silver_questions
    
    def generate_stress_questions(self, 
                                  all_documents: List,
                                  n_total: int = 10) -> List[TestQuestion]:
        """
        Genera domande STRESS dal corpus completo.
        Testa robustezza del sistema su domande "difficili".
        """
        from llama_index.core.schema import Document as LlamaDocument
        
        # Converti tutti i documenti
        llama_docs = [
            LlamaDocument(
                text=doc.page_content if hasattr(doc, 'page_content') else doc.get('content'),
                metadata=doc.metadata if hasattr(doc, 'metadata') else {},
                id_=f"doc_{i}"
            )
            for i, doc in enumerate(all_documents)
        ]
        
        testset = self.ragas_generator.generate_with_llamaindex_docs(
            documents=llama_docs,
            testset_size=n_total,
            raise_exceptions=False
        )
        
        stress_questions = []
        for i, example in enumerate(testset.to_list()):
            question = TestQuestion(
                question_id=f"stress_{i+1}",
                question=example.get('user_input', example.get('question')),
                tier=QuestionTier.STRESS,
                ground_truth_doc_ids=None,  # Non abbiamo ground truth per stress test
                reference_answer=example.get('reference')
            )
            stress_questions.append(question)
        
        return stress_questions
    
    def enrich_with_retrieval(self, questions: List[TestQuestion]) -> List[TestQuestion]:
        """
        Esegue il retriever su tutte le domande e calcola metriche.
        """
        print(f"\n🔍 Esecuzione retrieval su {len(questions)} domande...")
        
        for i, question in enumerate(questions):
            print(f"  [{i+1}/{len(questions)}] {question.tier.value}: {question.question[:50]}...")
            
            try:
                # Esegui il TUO retriever
                retrieved_docs = self.retrieve_fn(question.question)
                
                # Estrai info dai documenti recuperati
                retrieved_contexts = []
                retrieved_doc_ids = []
                
                for doc in retrieved_docs:
                    if isinstance(doc, dict):
                        content = doc.get('content', '')
                        doc_id = doc.get('id', doc.get('doc_id'))
                    elif hasattr(doc, 'page_content'):
                        content = doc.page_content
                        doc_id = doc.metadata.get('id', doc.metadata.get('doc_id'))
                    else:
                        content = str(doc)
                        doc_id = None
                    
                    retrieved_contexts.append(content)
                    if doc_id:
                        retrieved_doc_ids.append(str(doc_id))
                
                question.retrieved_contexts = retrieved_contexts
                question.retrieved_doc_ids = retrieved_doc_ids
                question.retrieved_count = len(retrieved_docs)
                
                # Calcola metriche se abbiamo ground truth
                if question.ground_truth_doc_ids:
                    metrics = self._calculate_metrics(
                        question.ground_truth_doc_ids,
                        retrieved_doc_ids
                    )
                    question.metrics = metrics
                    
                    print(f"      → Recuperati: {len(retrieved_docs)} docs")
                    print(f"      → Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}")
                
            except Exception as e:
                print(f"      ❌ Errore: {e}")
                question.retrieved_contexts = []
                question.retrieved_doc_ids = []
                question.retrieved_count = 0
        
        return questions
    
    def _calculate_metrics(self, 
                          ground_truth_ids: List[str], 
                          retrieved_ids: List[str]) -> Dict[str, float]:
        """
        Calcola Precision, Recall, F1 per una singola domanda.
        """
        gt_set = set(str(id) for id in ground_truth_ids)
        ret_set = set(str(id) for id in retrieved_ids)
        
        if not ret_set:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_set)}
        
        true_positives = len(gt_set & ret_set)
        false_positives = len(ret_set - gt_set)
        false_negatives = len(gt_set - ret_set)
        
        precision = true_positives / len(ret_set) if ret_set else 0
        recall = true_positives / len(gt_set) if gt_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def generate_full_testset(self,
                             n_gold_per_theme: int = 1,
                             n_silver: int = 10,
                             n_stress: int = 10,
                             all_documents: List = None,
                             manual_gold: bool = False) -> List[TestQuestion]:
        """
        Genera il testset completo stratificato.
        """
        print("\n" + "="*70)
        print("🎯 GENERAZIONE TESTSET IBRIDO STRATIFICATO")
        print("="*70)
        
        all_questions = []
        
        # 1. GOLD Questions
        print("\n📊 TIER 1: Gold Standard Questions (Ground Truth Verificato)")
        gold = self.generate_gold_questions(n_gold_per_theme, manual_gold)
        all_questions.extend(gold)
        print(f"✅ Generate {len(gold)} domande GOLD")
        
        # 2. SILVER Questions
        print("\n📊 TIER 2: Silver Standard Questions (Documenti Rilevanti)")
        silver = self.generate_silver_questions(n_silver)
        all_questions.extend(silver)
        print(f"✅ Generate {len(silver)} domande SILVER")
        
        # 3. STRESS Questions
        if all_documents:
            print("\n📊 TIER 3: Stress Test Questions (Corpus Completo)")
            stress = self.generate_stress_questions(all_documents, n_stress)
            all_questions.extend(stress)
            print(f"✅ Generate {len(stress)} domande STRESS")
        
        # 4. Retrieval + Metriche
        all_questions = self.enrich_with_retrieval(all_questions)
        
        return all_questions
    
    def save_testset(self, questions: List[TestQuestion], output_file: str):
        """Salva il testset in formato JSON strutturato"""
        testset_data = {
            'metadata': {
                'total_questions': len(questions),
                'gold_count': sum(1 for q in questions if q.tier == QuestionTier.GOLD),
                'silver_count': sum(1 for q in questions if q.tier == QuestionTier.SILVER),
                'stress_count': sum(1 for q in questions if q.tier == QuestionTier.STRESS),
                'themes': list(set(q.theme_id for q in questions if q.theme_id))
            },
            'questions': [q.to_dict() for q in questions]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(testset_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Testset salvato in: {output_file}")
        self._print_statistics(questions)
    
    def _print_statistics(self, questions: List[TestQuestion]):
        """Stampa statistiche dettagliate del testset"""
        print("\n📊 STATISTICHE TESTSET:")
        print("="*70)
        
        # Per tier
        for tier in QuestionTier:
            tier_questions = [q for q in questions if q.tier == tier]
            if tier_questions:
                print(f"\n{tier.value.upper()}:")
                print(f"  • Numero domande: {len(tier_questions)}")
                
                # Metriche medie (solo per GOLD e SILVER con ground truth)
                with_metrics = [q for q in tier_questions if q.metrics]
                if with_metrics:
                    avg_precision = sum(q.metrics['precision'] for q in with_metrics) / len(with_metrics)
                    avg_recall = sum(q.metrics['recall'] for q in with_metrics) / len(with_metrics)
                    avg_f1 = sum(q.metrics['f1'] for q in with_metrics) / len(with_metrics)
                    
                    print(f"  • Precision media: {avg_precision:.3f}")
                    print(f"  • Recall media: {avg_recall:.3f}")
                    print(f"  • F1 Score medio: {avg_f1:.3f}")
                
                avg_retrieved = sum(q.retrieved_count for q in tier_questions) / len(tier_questions)
                print(f"  • Documenti recuperati (media): {avg_retrieved:.1f}")


# ============================================================================
# ESEMPIO DI UTILIZZO
# ============================================================================

def example_usage():
    """
    Esempio completo di utilizzo del generatore ibrido.
    Adatta questo al tuo caso specifico.
    """
    
    # 1. Definisci i tuoi 10 domini tematici
    domains = [
        ThematicDomain(
            theme_id="gestione_magazzino",
            theme_name="Gestione Magazzino",
            theme_description="Configurazione e operazioni sui magazzini in SAM ERP2",
            reference_docs=[
                GroundTruthDocument(
                    doc_id="RI_1234_chunk_1",
                    content="...",
                    metadata={"NumRI": 1234, "Cliente": "ClienteX"},
                    relevance_score=1.0
                ),
                # ... altri 3-7 documenti
            ]
        ),
        # ... altri 9 temi
    ]
    
    # 2. Inizializza il generatore
    # (Usa i tuoi componenti esistenti)
    from your_module import get_ragas_generator, retrieve_fn
    
    generator = HybridTestsetGenerator(
        thematic_domains=domains,
        ragas_generator=get_ragas_generator(),
        retrieve_fn=retrieve_fn
    )
    
    # 3. Genera il testset completo
    questions = generator.generate_full_testset(
        n_gold_per_theme=1,      # 1 domanda gold per tema = 10 totali
        n_silver=10,              # 10 domande silver
        n_stress=10,              # 10 domande stress
        all_documents=load_all_docs(),
        manual_gold=False         # True per inserimento manuale (più qualità)
    )
    
    # 4. Salva
    generator.save_testset(questions, "testset_ibrido_stratificato.json")
    
    return questions


if __name__ == "__main__":
    # Esegui esempio
    questions = example_usage()