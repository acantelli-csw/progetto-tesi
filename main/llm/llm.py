from dotenv import load_dotenv
from openai import AzureOpenAI
import json
import os
import search
import tiktoken

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
    un sistema di ricerca. Devi identificare quali, tra i documenti estratti, sono quelli 
    strettamente utili per rispondere a una richiesta utente e 
    coerenti con essa nello specifico e quali no.
    Questo perchè la ricerca documentale fornisce sempre i top 10 documenti più rilevanti 
    ma non è detto che siano sempre tutti effettivamente utili. Per ciascun documento viene fornito, 
    oltre al contenuto, anche il valore di similarità coseno del suo embedding con quello del 
    prompt utente. Puoi sfruttare anche queste informazioni per filtrare i documenti 
    (ad esempio per valori notevolmente più bassi degli altri).

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
                {"titolo": "Report vendite mensili", "content": "Contiene dati di vendita per cliente e prodotto.", "autore": "Mario", "cliente": "ClienteA", "similarity": 0.42},
                {"titolo": "Movimenti di magazzino", "content": "Elenco dettagliato dei movimenti di magazzino con filtri per data e prodotto.", "autore": "Luigi", "cliente": "ClienteB", "similarity": 0.91},
                {"titolo": "Guida configurazione magazzino", "content": "Istruzioni su come configurare il modulo magazzino nel software.", "autore": "Anna", "cliente": "ClienteC", "similarity": 0.75}
            ],
            "decision": {
                "relevant_docs": [1, 2],
                "irrelevant_docs": [0],
                "reason": "Il secondo documento è perfettamente coerente con la richiesta; il terzo è utile come supporto tecnico. Il primo documento ha similarità bassa e non contiene informazioni sui movimenti di magazzino."
            }
        },
        {
            "prompt": "Come posso impostare le chiavi univoche per i clienti?",
            "documents": [
                {"titolo": "Chiavi univoche clienti", "content": "Spiega come configurare chiavi univoche per evitare duplicazioni di dati cliente.", "autore": "Luca", "cliente": "ClienteX", "similarity": 0.95},
                {"titolo": "Gestione magazzini", "content": "Contiene istruzioni su come creare nuovi magazzini e assegnare codici identificativi.", "autore": "Marco", "cliente": "ClienteY", "similarity": 0.55},
                {"titolo": "Parametri generali azienda", "content": "Descrive le impostazioni generali dell'azienda, ma non tratta le chiavi univoche.", "autore": "Elisa", "cliente": "ClienteZ", "similarity": 0.40}
            ],
            "decision": {
                "relevant_docs": [0],
                "irrelevant_docs": [1, 2],
                "reason": "Solo il primo documento tratta direttamente le chiavi univoche. Gli altri hanno similarità più bassa e non forniscono informazioni utili per la richiesta."
            }
        },
        {
            "prompt": "Perché ricevo l'errore 'magazzino non trovato' quando salvo un ordine?",
            "documents": [
                {"titolo": "Errori comuni nella gestione ordini", "content": "L'errore 'magazzino non trovato' si verifica quando il magazzino indicato non è attivo o non è associato all'azienda selezionata.", "autore": "Verdi", "cliente": "Cliente1", "similarity": 0.92},
                {"titolo": "Configurazione magazzino", "content": "Descrive come attivare un magazzino e associarlo a un'azienda dal menu Anagrafica Magazzini.", "autore": "Rossi", "cliente": "Cliente2", "similarity": 0.88},
                {"titolo": "Report vendite annuali", "content": "Analisi delle vendite su base annuale con filtri per categoria e zona geografica.", "autore": "Neri", "cliente": "Cliente3", "similarity": 0.33}
            ],
            "decision": {
                "relevant_docs": [0, 1],
                "irrelevant_docs": [2],
                "reason": "I primi due documenti trattano direttamente l'errore e la sua risoluzione. Il terzo documento ha bassa similarità e non è utile per il problema."
            }
        },
        {
            "prompt": "Come posso generare un report dei movimenti di magazzino?",
            "documents": [
                {"titolo": "Report vendite mensili", "content": "Contiene dati di vendita per cliente e prodotto.", "autore": "Mario", "cliente": "ClienteA", "similarity": 0.55},
                {"titolo": "Movimenti di magazzino", "content": "Elenco dettagliato dei movimenti di magazzino con filtri per data e prodotto.", "autore": "Luigi", "cliente": "ClienteB", "similarity": 0.76},
                {"titolo": "Guida configurazione magazzino", "content": "Istruzioni su come configurare il modulo magazzino nel software.", "autore": "Anna", "cliente": "ClienteC", "similarity": 0.74}
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
                {"titolo": "Fatture clienti", "content": "Guida alla gestione fatture e modifica dati cliente.", "autore": "Andrea", "cliente": "ClienteD", "similarity": 0.60},
                {"titolo": "Gestione magazzino", "content": "Movimenti di magazzino e scorte.", "autore": "Riccardo", "cliente": "ClienteE", "similarity": 0.42}
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

def generate_final_answer(user_prompt: str, selected_docs: list, chat_history: list = None) -> str:

    context_text = "\n\n".join([
        f"Numero RI: {d['numero']} - Titolo: {d['titolo']}, Chunk: {d['progressivo']}, Autore: {d['autore']}, Cliente: {d['cliente']}\n - Contenuto: {d['content']}" 
        for d in selected_docs]) if selected_docs else "Nessun documento rilevante trovato."

    system_message = """
    Sei un assistente esperto di un software gestionale. Devi rispondere alle richieste degli utenti usando solo le informazioni presenti nei documenti forniti. 
    Regole:
    1. Non inventare informazioni o dettagli che non sono nei documenti.
    2. Annotare ogni riferimento a un documento con un numero tra parentesi quadre [1], [2], ecc.
    3. Alla fine della risposta, fornire la lista dei documenti di riferimento usati. In questo formato ('numero'=41699 nell'esempio): 1. RI: [41699](https://intranet.centrosoftware.com/IntraCSW/script/vedi_RI.asp?idRI=41699) - 'titolo', Chunk: 'progressivo+1', Autore: 'autore', Cliente 'cliente'.
    4. Mantieni la risposta chiara e strutturata, basata esclusivamente sui documenti forniti.
    5. Se non vengono forniti documenti rispondi in modo naturale e umano, come se fossi un chatbot.
    6. In ogni caso, genera l'oupput in formato markdown, così che possa essere utilizzato direttamente all'interno di un'interfaccia Streamlit.
    """

    examples = [
        {
            "user_prompt": "Come posso creare un nuovo ordine cliente nel gestionale?",
            "documents": [
                {
                    "numero": "41699",
                    "titolo": "Creazione Ordine Cliente",
                    "progressivo": "2",
                    "content": "Per creare un nuovo ordine cliente, accedere al modulo Vendite > Ordini > Nuovo. Compilare i campi obbligatori: Cliente, Data, Magazzino e Condizioni di pagamento. Salvare per confermare. È possibile generare il documento PDF dal pulsante 'Stampa'.",
                    "autore": "Rossi",
                    "cliente": "Centro Software"
                },
                {
                    "numero": "41700",
                    "titolo": "Gestione anagrafiche clienti",
                    "progressivo": "0",
                    "content": "Le anagrafiche clienti devono essere create prima dell'inserimento di ordini o documenti di vendita.",
                    "autore": "Bianchi",
                    "cliente": "Centro Software"
                }
            ],
            "answer": "Per creare un nuovo ordine cliente, accedi al modulo **Vendite → Ordini → Nuovo**. Compila i campi obbligatori (Cliente, Data, Magazzino e Condizioni di pagamento) e salva per confermare [1]. Ricorda che il cliente deve essere già presente in anagrafica prima di procedere [2].\n\nDocumenti di riferimento:\n1. RI: [41699](https://intranet.centrosoftware.com/IntraCSW/script/vedi_RI.asp?idRI=41699) - Creazione Ordine Cliente, Chunk: 3, Autore: Rossi, Cliente: Centro Software\n2. RI: [41700](https://intranet.centrosoftware.com/IntraCSW/script/vedi_RI.asp?idRI=41700) - Gestione anagrafiche clienti, Chunk: 1, Autore: Bianchi, Cliente: Centro Software"
        },
        {
            "user_prompt": "Cosa significa l'errore 'magazzino non trovato' durante il salvataggio di un ordine?",
            "documents": [
                {
                    "numero": "42210",
                    "titolo": "Errori comuni nella gestione ordini",
                    "progressivo": "6",
                    "content": "L'errore 'magazzino non trovato' si verifica quando il magazzino indicato nell'ordine non è attivo o non è associato all'azienda selezionata.",
                    "autore": "Verdi",
                    "cliente": "Centro Software"
                },
                {
                    "numero": "42211",
                    "titolo": "Configurazione magazzini",
                    "progressivo": "1",
                    "content": "Per verificare i magazzini attivi, accedere al menu Magazzini > Anagrafica. Attivare il flag 'Attivo' per renderli disponibili nei documenti di vendita.",
                    "autore": "Neri",
                    "cliente": "Centro Software"
                }
            ],
            "answer": "L'errore **'magazzino non trovato'** indica che il magazzino selezionato non è attivo o non è associato all'azienda corrente [1]. Per risolvere, vai in **Magazzini → Anagrafica** e verifica che il magazzino sia attivo (flag 'Attivo' selezionato) e collegato all'azienda corretta [2].\n\nDocumenti di riferimento:\n1. RI: [42210](https://intranet.centrosoftware.com/IntraCSW/script/vedi_RI.asp?idRI=42210) - Errori comuni nella gestione ordini, Chunk: 7, Autore: Verdi, Cliente: Centro Software\n2. RI: [42211](https://intranet.centrosoftware.com/IntraCSW/script/vedi_RI.asp?idRI=42211) - Configurazione magazzini, Chunk: 2   , Autore: Neri, Cliente: Centro Software"
        },
        {
            "user_prompt": "Qual è la politica aziendale per le ferie dei dipendenti?",
            "documents": [],
            "answer": "Mi dispiace, ma non ho trovato nessuna informazione sui criteri di gestione delle ferie dei dipendenti nei documenti disponibili. Posso però aiutarti a indirizzare la richiesta al reparto Risorse Umane o fornirti indicazioni generali se mi dai maggiori dettagli."
        },
        {
            "user_prompt": "Come configurare i piani di consegna e le chiavi univoche?",
            "documents": [
                {
                    "numero": "12121",
                    "titolo": "Configurazione piani di consegna",
                    "progressivo": "4",
                    "content": "Per impostare i piani di consegna, definire importazione e gestione dei dati.",
                    "autore": "Mario",
                    "cliente": "Alfa"
                },
                {
                    "numero": "12345",
                    "titolo": "Gestione chiavi univoche",
                    "progressivo": "0",
                    "content": "Le chiavi univoche devono essere definite per ogni cliente per evitare conflitti.",
                    "autore": "Luca",
                    "cliente": "Beta"}
            ],
            "answer": "Per configurare i piani di consegna, impostare importazione e gestione dei dati [1]. Le chiavi univoche devono essere definite per ciascun cliente [2].\n\nDocumenti di riferimento:\n1. RI: [12121](https://intranet.centrosoftware.com/IntraCSW/script/vedi_RI.asp?idRI=12121) - Configurazione piani di consegna, Chunk: 5, Autore: Mario, Cliente: Alfa\n2. RI: [12345](https://intranet.centrosoftware.com/IntraCSW/script/vedi_RI.asp?idRI=12345) - Gestione chiavi univoche, Chunk: 1, Autore: Luca, Cliente: Beta"
        },
        {
            "user_prompt": "Come verificare i dati importati nel sistema?",
            "documents": [
                {
                    "numero": "51000",
                    "titolo": "Controllo dati importati",
                    "progressivo": "3",
                    "content": "Verificare che tutti i campi siano completi e corretti dopo l'importazione.",
                    "autore": "Anna",
                    "cliente": "Gamma"}
            ],
            "answer": "Per verificare i dati importati, controllare che tutti i campi siano completi e corretti [1].\n\nDocumenti di riferimento:\n1. RI: [51000](https://intranet.centrosoftware.com/IntraCSW/script/vedi_RI.asp?idRI=51000) - Controllo dati importati, Chunk: 4, Autore: Anna, Cliente: Gamma"
        }
    ]

    few_shot_text = ""
    for ex in examples:
        docs_text_example = "\n".join([f"{i+1}: Numero RI: {d['numero']} - Titolo: {d['titolo']}, Autore: {d['autore']}, Cliente: {d['cliente']}\n - Contenuto: {d['content']}" for i, d in enumerate(ex['documents'])
        ])
        few_shot_text += f"Prompt utente: {ex['user_prompt']}\nDocumenti:\n{docs_text_example}\nRisposta attesa:\n{ex['answer']}\n\n"

    summarized_chat = summarize_chat_history(chat_history)

    user_message = f"""
    {few_shot_text}

    Contesto conversazionale recente:
    {summarized_chat}

    Prompt utente:
    {user_prompt}

    Documenti disponibili:
    {context_text}

    Risposta attesa:
    """

    response = client.chat.completions.create(
        model = MODEL_NAME,
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
    )

    return response.choices[0].message.content


# FLUSSO COMPLETO DEL CHATBOT =====================================================

def gpt_request(messages):

    # Estrae l'ultimo prompt inserito dalla cronologia chat
    user_prompt = [m["content"] for m in messages if m["role"] == "user"][-1]

    # 1️ - Decisione strumenti
    tools = decide_tools(user_prompt)
    print("\nRagionamento sugli strumenti da usare:")
    print(tools["reason"])

    # 1.5 - Recupero documenti
    all_documents = []
    if tools["use_semantic"]:
        print("\nUso la ricerca SEMANTICA\n")
        all_documents = search.semantic_search(user_prompt)

    if tools["use_keyword"]:
        print("\nUso la ricerca per KEYWORDS (ancora da implementare)\n")
        all_documents += search.keyword_search(user_prompt)

    # 2 - Selezione documenti in base alla coerenza
    document_selection = []
    selected_docs = []
    if tools["use_semantic"] or tools["use_keyword"]:
        if all_documents:
            document_selection = select_documents(user_prompt, all_documents)

            # Selezione documenti rilevanti
            selected_docs = [all_documents[i] for i in document_selection['relevant_docs']]

    # 3 - Genera risposta finale
    final_answer = generate_final_answer(user_prompt, selected_docs, messages)

    return final_answer


# GESTIONE LUNGHEZZA CRONOLOGIA MESSAGGI =====================================================
def summarize_chat_history(chat_history, model_name=MODEL_NAME):

    if not chat_history:
        return ""
    
    encoding = tiktoken.encoding_for_model(model_name)
    max_tokens=3000
    reserved_tokens_for_summary = 500
    allowed_recent_tokens = max_tokens - reserved_tokens_for_summary

    # Se la chat è più corta del limite, restituisci tutta la chat senza ulteriori elaborazioni
    total_chat_tokens = sum(len(encoding.encode(f"{'Utente' if m['role']=='user' else 'Assistente'}: {m['content']}\n")) for m in chat_history)

    if total_chat_tokens <= max_tokens:
        return "".join(f"{'Utente' if m['role']=='user' else 'Assistente'}: {m['content']}\n" for m in chat_history)
    
    recent_lines = []
    recent_tokens = 0

    # Salvo messaggi più recenti
    for m in reversed(chat_history):
        role = "Utente" if m["role"] == "user" else "Assistente"
        line = f"{role}: {m['content']}\n"
        line_tokens = len(encoding.encode(line))
        if recent_tokens + line_tokens <= allowed_recent_tokens:
            recent_lines.insert(0, line)  # inserisci in cima per mantenere ordine cronologico
            recent_tokens += line_tokens
        else:
            break  # limite superato
 
    # Riassumo messaggi vecchi
    old_messages = chat_history[:len(chat_history) - len(recent_lines)]
    summary_text = ""
    if old_messages:
        summary = summarize_old_messages(old_messages, reserved_tokens_for_summary)
        summary_text = f"Riassunto messaggi precedenti: {summary}\n"

    return summary_text + "".join(recent_lines)

# RIASSUNTO MESSAGGI VECCHI
def summarize_old_messages(messages, max_tokens):

    model_name=MODEL_NAME

    if not messages:
        return ""
    
    system_prompt = """Sei un assistente che deve riassumere i messaggi precedenti della chat
                        in modo sintetico, mantenendo solo i concetti principali.
                        Il riassunto deve essere chiaro, breve e utile al contesto"""

    # Costruisci il testo
    text_to_summarize = ""
    for m in messages:
        role = "Utente" if m["role"] == "user" else "Assistente"
        text_to_summarize += f"{role}: {m['content']}\n"
    
    # Chimata LLM per riassunto
    response = client.chat.completions.create(
        model = model_name,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_to_summarize}
        ],
        max_tokens=max_tokens
    )

    return response.choices[0].message.content