from dotenv import load_dotenv
from openai import AzureOpenAI
import json
import os
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


# 1. DECISIONE TOOLS DA USARE =====================================================
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
# TODO: add examples and tune the system prompt

def select_documents(user_prompt: str, documents: list) -> dict:
    
    system_message = """
    Sei un assistente decisionale incaricato di filtrare documenti estratti da
    un sistema di ricerca. Devi identificare quali, tra i documenti estratti, sono quelli 
    strettamente utili per rispondere a una richiesta utente e 
    coerenti con essa nello specifico e quali no.

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
                {"titolo": "Report vendite mensili", "content": "Contiene dati di vendita per cliente e prodotto.", "autore": "Mario", "cliente": "ClienteA"},
                {"titolo": "Movimenti di magazzino", "content": "Elenco dettagliato dei movimenti di magazzino con filtri per data e prodotto.", "autore": "Luigi", "cliente": "ClienteB"},
                {"titolo": "Guida configurazione magazzino", "content": "Istruzioni su come configurare il modulo magazzino nel software.", "autore": "Anna", "cliente": "ClienteC"}
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
                {"titolo": "Fatture clienti", "content": "Guida alla gestione fatture e modifica dati cliente.", "autore": "Andrea", "cliente": "ClienteD"},
                {"titolo": "Gestione magazzino", "content": "Movimenti di magazzino e scorte.", "autore": "Riccardo", "cliente": "ClienteE"}
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
        docs_text = "\n".join([f"{i}: {d['titolo']} - {d['content']} (autore: {d['autore']}, cliente: {d['cliente']})" for i, d in enumerate(ex['documents'])])
        few_shot_text += f"Prompt utente: {ex['prompt']}\nDocumenti:\n{docs_text}\nDecisione: {json.dumps(ex['decision'])}\n\n"

    # Testo del prompt per il modello
    docs_text = "\n".join([f"{i}: {d['titolo']} - {d['content']} (autore: {d['autore']}, cliente: {d['cliente']})" for i, d in enumerate(documents)])
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

def generate_final_answer(user_prompt: str, selected_docs: list) -> str:

    context_text = "\n\n".join([f" Titolo: {d['titolo']}, Autore: {d['autore']}, Cliente: {d['cliente']}\n - Contenuto: {d['content']}" for d in selected_docs]) if selected_docs else "Nessun documento rilevante trovato."

    system_message = """
    Sei un assistente esperto di un software gestionale. Devi rispondere alle richieste degli utenti usando solo le informazioni presenti nei documenti forniti. 
    Regole:
    1. Non inventare informazioni o dettagli che non sono nei documenti.
    2. Annotare ogni riferimento a un documento con un numero tra parentesi quadre [1], [2], ecc.
    3. Alla fine della risposta, fornire la lista dei documenti di riferimento usati.
    4. Mantieni la risposta chiara e strutturata, basata esclusivamente sui documenti forniti.
    """

    examples = [
        {
            "user_prompt": "Come configurare i piani di consegna e le chiavi univoche?",
            "documents": [
                {"titolo": "Configurazione piani di consegna", "content": "Per impostare i piani di consegna, definire importazione e gestione dei dati.", "autore": "Mario", "cliente": "Alfa"},
                {"titolo": "Gestione chiavi univoche", "content": "Le chiavi univoche devono essere definite per ogni cliente per evitare conflitti.", "autore": "Luca", "cliente": "Beta"}
            ],
            "answer": "Per configurare i piani di consegna, impostare importazione e gestione dei dati [1]. Le chiavi univoche devono essere definite per ciascun cliente [2].\n\nDocumenti di riferimento:\n1. Titolo: Configurazione piani di consegna, Autore: Mario, Cliente: Alfa\n2. Titolo: Gestione chiavi univoche, Autore: Luca, Cliente: Beta"
        },
        {
            "user_prompt": "Come verificare i dati importati nel sistema?",
            "documents": [
                {"titolo": "Controllo dati importati", "content": "Verificare che tutti i campi siano completi e corretti dopo l'importazione.", "autore": "Anna", "cliente": "Gamma"}
            ],
            "answer": "Per verificare i dati importati, controllare che tutti i campi siano completi e corretti [1].\n\nDocumenti di riferimento:\n1. Titolo: Controllo dati importati, Autore: Anna, Cliente: Gamma"
        }
    ]

        # Costruzione del testo degli esempi da dare al modello
    
    few_shot_text = ""
    for ex in examples:
        docs_text_example = "\n".join([f"{i+1}: Titolo: {d['titolo']}, Autore: {d['autore']}, Cliente: {d['cliente']}\n - Contenuto: {d['content']}" for i, d in enumerate(ex['documents'])
        ])
        few_shot_text += f"Prompt utente: {ex['user_prompt']}\nDocumenti:\n{docs_text_example}\nRisposta attesa:\n{ex['answer']}\n\n"

    user_message = f"""
    {few_shot_text}

    Prompt utente: {user_prompt}

    Documenti disponibili:
    {context_text}

    Risposta attesa:
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0
    )

    return response.choices[0].message.content


# FLUSSO COMPLETO DEL CHATBOT =====================================================

def gpt_request(user_prompt):
# TODO use output_line from semantic search (es. filtra quelli con similirità sotto la media)

    # 1️ - Decisione strumenti
    tools = decide_tools(user_prompt)
    print("\nRagionamento sugli strumenti da usare:")
    print(tools["reason"])

    # 1.5 - Recupero documenti
    all_documents = []
    if tools["use_semantic"]:
        print("\nUso la ricerca SEMANTICA\n")
        all_documents, output_line = search.semantic_search(user_prompt)

    if tools["use_keyword"]:
        print("\nUso la ricerca per KEYWORDS (ancora da implementare)\n")
        # all_documents += search.keyword_search(user_prompt)

    # 2 - Selezione documenti in base alla coerenza
    document_selection = []
    selected_docs = []
    if tools["use_semantic"] or tools["use_keyword"]:
        if all_documents:
            document_selection = select_documents(user_prompt, all_documents)

            # Selezione documenti rilevanti
            selected_docs = [all_documents[i] for i in document_selection['relevant_docs']]

    # 3 - Genera risposta finale
    final_answer = generate_final_answer(user_prompt, selected_docs)
    return final_answer



