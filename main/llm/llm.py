import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
import search


# 0. CONFIGURAZIONE ==================================================

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


# 1. DECISIONE TOOLS ========================================
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

# ========================================
# 3. SELEZIONE DOCUMENTI
# ========================================

def select_documents(user_prompt, documents):

    if not documents:
        return []

    system_prompt = (
        "Sei un assistente intelligente. Hai una lista di documenti o chunk di documenti estratti da un database. "
        "Il tuo compito è selezionare solo i documenti coerenti e rilevanti rispetto alla domanda dell'utente."
    )

    user_content = (
        f"Domanda utente: {user_prompt}\n\n"
        "Documenti estratti:\n" +
        "\n".join(f"{i+1}. {doc}" for i, doc in enumerate(documents)) +
        "\n\nRestituisci solo i numeri dei documenti rilevanti in una lista JSON, ad esempio: [1, 3, 5]"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )

    # Parsing della lista dei documenti rilevanti
    try:
        relevant_indices = json.loads(response.choices[0].message["content"])
        filtered_docs = [documents[i-1] for i in relevant_indices if 0 < i <= len(documents)]
        return filtered_docs
    except Exception as e:
        print("Errore nel parsing del filtro documenti:", e)
        return documents  # fallback: restituisci tutti i documenti


# ========================================
# 4. GENERAZIONE RISPOSTA FINALE
# ========================================

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


# ========================================
# FLUSSO COMPLETO DEL CHATBOT
# ========================================

def gpt_request(user_prompt):

    # 1️. Decisione strumenti
    tools = decide_tools(user_prompt)
    print(tools)
    
    # 2️. Recupero documenti
    if tools["use_semantic_search"]:
        documents, output_line = search.semantic_search(user_prompt)
    # if tools["use_keyword_search"]:
        # documents += search.keyword_search(user_prompt)

    # 3. Selezione documenti in base alla coerenza
    best_documents = []
    if documents:
        best_documents = select_documents(user_prompt, documents)

    # 4. Genera risposta finale
    final_response = generate_final_response(user_prompt, best_documents)
    return final_response



