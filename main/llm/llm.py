import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
import search


# 0. CONFIGURAZIONE =====================================================

# Carica variabili d'ambiente
load_dotenv()

# Inizializza il client Azure OpenAI
client = AzureOpenAI(
    azure_endpoint=os.getenv("LLM_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("LLM_VERSION")
)

MODEL_NAME = os.getenv("LLM_MODEL")

# Lettura del systme prompt da file
prompt_path = Path(__file__).parent / "prompt.txt"
SYSTEM_PROMPT = prompt_path.read_text(encoding="utf-8")


# 1. DECISIONE TOOLS =====================================================
# TODO: tune and add examples for better performance

def decide_tools(prompt: str) -> dict:

    system_message = """
    Sei un assistente decisionale. Devi rispondere SOLO con una decisione su
    quali strumenti di ricerca usare per un prompt utente, in base alla
    pertinenza ai documenti di richieste di implementazione clienti
    per un software gestionale aziendale.

    I due sistemi di ricerca:
    1. Ricerca semantica: per concetti generali, complessi o combinazioni di funzionalità.
    2. Ricerca per keyword: per termini specifici, nomi di moduli o funzionalità precise.

    Rispondi SEMPRE in formato JSON:
    {
        "use_semantic": true/false,
        "use_keyword": true/false,
        "reason": "<spiegazione breve della decisione>"
    }
    """

    examples = [
        {
            "prompt": "Come generare un report dettagliato delle vendite per cliente?",
            "decision": {
                "use_semantic": True,
                "use_keyword": False,
                "reason": "Richiesta di report generali, utile usare ricerca semantica."
            }
        },
        {
            "prompt": "Elenca fatture inviate ad un cliente specifico.",
            "decision": {
                "use_semantic": False,
                "use_keyword": True,
                "reason": "Richiesta precisa su documenti specifici, meglio ricerca per keyword."
            }
        },
        {
            "prompt": "Implementazione mrp",
            "decision": {
                "use_semantic": True,
                "use_keyword": True,
                "reason": "Richiesta generica ma include termini specifici dei moduli."
            }
        },
        {
            "prompt": "Configurazione di un modulo di magazzino e contabilità?",
            "decision": {
                "use_semantic": True,
                "use_keyword": True,
                "reason": "Richiesta generica ma include termini specifici dei moduli."
            }
        },
        {
            "prompt": "Quali sono le migliori tecniche di gestione del tempo?",
            "decision": {
                "use_semantic": False,
                "use_keyword": False,
                "reason": "Non riguarda i documenti o funzionalità del software gestionale."
            }
        }
    ]

    # Costruisco il prompt few-shot
    few_shot_text = ""
    for ex in examples:
        few_shot_text += f"Prompt utente: {ex['prompt']}\nDecisione: {json.dumps(ex['decision'])}\n\n"

    user_message = f"{few_shot_text}Prompt utente: {prompt}\nDecisione:"

    response = client.chat.completions.create(
        model = MODEL_NAME,
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0
    )
    
    decision_text = response.choices[0].message.content
    
    try:
        decision = json.loads(decision_text)
    except json.JSONDecodeError:
        # fallback se il modello non restituisce JSON valido
        decision = {
            "use_semantic": False,
            "use_keyword": False,
            "reason": f"Errore nel parsing della risposta: {decision_text}"
        }
    
    return decision


# 2. SELEZIONE DOCUMENTI =====================================================

def select_documents(user_prompt: str, documents: list) -> dict:
    
    system_message = """
    Sei un assistente decisionale incaricato di filtrare documenti estratti da
    un sistema di ricerca. Devi identificare quali documenti sono effettivamente
    utili per rispondere a una richiesta utente e quali no.

    Criteri:
    - Il documento è rilevante se contiene informazioni utili per rispondere
      al prompt dell'utente.
    - Alcuni documenti potrebbero essere estratti solo perché moderatamente
      coerenti, ma non aiutano concretamente nella risposta.
    - Rispondi SEMPRE in formato JSON con gli indici dei documenti:
      {
          "relevant_docs": [indici dei documenti utili],
          "irrelevant_docs": [indici dei documenti non utili],
          "reason": "<breve spiegazione>"
      }
    """

    examples = [
        {
            "prompt": "Come posso generare un report dei movimenti di magazzino?",
            "documents": [
                {"title": "Report vendite mensili", "content": "Contiene dati di vendita per cliente e prodotto."},
                {"title": "Movimenti di magazzino", "content": "Elenco dettagliato dei movimenti di magazzino con filtri per data e prodotto."},
                {"title": "Guida configurazione magazzino", "content": "Istruzioni su come configurare il modulo magazzino nel software."}
            ],
            "decision": {
                "relevant_docs": [1, 2],
                "irrelevant_docs": [0],
                "reason": "Il secondo e terzo documento contengono informazioni direttamente utili al report richiesto."
            }
        },
        {
            "prompt": "Come posso modificare la fattura di un cliente?",
            "documents": [
                {"title": "Fatture clienti", "content": "Guida alla gestione fatture e modifica dati cliente."},
                {"title": "Gestione magazzino", "content": "Movimenti di magazzino e scorte."}
            ],
            "decision": {
                "relevant_docs": [0],
                "irrelevant_docs": [1],
                "reason": "Solo il primo documento riguarda la modifica delle fatture."
            }
        }
    ]

    # Costruisco il few-shot text
    few_shot_text = ""
    for ex in examples:
        docs_text = "\n".join([f"{i}: {d['title']} - {d['content']}" for i, d in enumerate(ex['documents'])])
        few_shot_text += f"Prompt utente: {ex['prompt']}\nDocumenti:\n{docs_text}\nDecisione: {json.dumps(ex['decision'])}\n\n"

    # Testo del prompt per il modello
    docs_text = "\n".join([f"{i}: {d['title']} - {d['content']}" for i, d in enumerate(documents)])
    user_message = f"{few_shot_text}Prompt utente: {user_prompt}\nDocumenti:\n{docs_text}\nDecisione:"

    # Chiamata al modello
    response = client.chat.completions.create(
        model = MODEL_NAME,
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0
    )

    decision_text = response.choices[0].message.content

    # Parsing JSON con fallback
    try:
        decision = json.loads(decision_text)
    except json.JSONDecodeError:
        decision = {
            "relevant_docs": [],
            "irrelevant_docs": list(range(len(documents))),
            "reason": f"Errore nel parsing della risposta: {decision_text}"
        }

    return decision

# 3. GENERAZIONE RISPOSTA FINALE =====================================================

def generate_final_response(user_prompt, documents):

    context_text = "\n\n".join(documents) if documents else "Nessun documento rilevante trovato."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Domanda: {user_prompt}\n\nDocumenti:\n{context_text}"}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )

    return response.choices[0].message["content"]


# FLUSSO COMPLETO DEL CHATBOT =====================================================

def gpt_request(user_prompt):
# TODO improve and use output_line from semantic search

    # 1️ - Decisione strumenti
    tools = decide_tools(user_prompt)
    print("\nRagionamento sugli strumenti da usare:")
    print(tools["reason"])

    # 1.5 - Recupero documenti
    if tools["use_semantic"]:
        print("\nUso la ricerca SEMANTICA\n")
        documents, output_line = search.semantic_search(user_prompt)

    if tools["use_keyword"]:
        print("\nUso la ricerca per KEYWORDS (ancora da implementare)\n")
        # documents += search.keyword_search(user_prompt)

    # FINO QUI TUTTO OK

    # 2 - Selezione documenti in base alla coerenza
    # TODO test this one
    best_documents = []
    if documents:
        best_documents = select_documents(user_prompt, documents)

    # 3 - Genera risposta finale
    final_response = generate_final_response(user_prompt, best_documents)
    return final_response



