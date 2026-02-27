from dotenv import load_dotenv
from openai import AzureOpenAI
import json
import os
from search import semantic_search, keyword_search
import tiktoken
from docx import Document
from difflib import SequenceMatcher

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

    examples = [
        {
            "prompt": "Come generare un report dettagliato delle vendite per cliente?",
            "decision": {
                "use_semantic": True,
                "use_keyword": False,
                "reason": "Richiesta generica su funzionalità standard di SAM ERP2 (report vendite). La ricerca semantica aiuta a trovare documenti con esempi di report, anche da plugin correlati."
            }
        },
        {
            "prompt": "Elenca fatture inviate ad un cliente specifico.",
            "decision": {
                "use_semantic": False,
                "use_keyword": True,
                "reason": "Richiesta precisa su documenti standard di SAM ERP2 (fatture cliente). La ricerca per keyword trova velocemente riferimenti a moduli contabilità e fatturazione specifici."
            }
        },
        {
            "prompt": "Implementazione personalizzata di MRP per un cliente con esigenze particolari",
            "decision": {
                "use_semantic": True,
                "use_keyword": True,
                "reason": "Richiesta su plugin o personalizzazioni MRP. La ricerca semantica aiuta a trovare concetti correlati, la keyword trova riferimenti a RI specifiche già sviluppate."
            }
        },
        {
            "prompt": "Configurazione di un modulo di magazzino e contabilità in SAM ERP2",
            "decision": {
                "use_semantic": True,
                "use_keyword": True,
                "reason": "Richiesta generica che include moduli standard, ma anche possibili personalizzazioni: entrambe le ricerche possono essere utili."
            }
        },
        {
            "prompt": "Quali sono le migliori tecniche di gestione del tempo per i dipendenti?",
            "decision": {
                "use_semantic": False,
                "use_keyword": False,
                "reason": "Non riguarda funzionalità standard né plugin di SAM ERP2, quindi i documenti RI non sono rilevanti."
            }
        }
    ]

    # Costruisco il few-shot prompt 
    few_shot_text = ""
    for ex in examples:
        few_shot_text += f"Prompt utente: {ex['prompt']}\nDecisione: {json.dumps(ex['decision'])}\n\n"

    # STEP 1
    system_message = f"""
    Sei un assistente decisionale esperto del software gestionale SAM ERP2 dell'azienda Centro Software. 
    Devi rispondere SOLO con una decisione su quali strumenti di ricerca usare per un prompt utente, 
    basandoti sul tipo di richiesta e sulla pertinenza rispetto ai documenti contenenti le Richieste di Implementazione (RI) fatte dai clienti.

    Considera quanto segue:
    - Il software SAM ERP2 ha funzionalità standard note, e il modello conosce queste funzionalità.
    - Le personalizzazioni e i plugin richiesti dai clienti sono invece contenuti nei documenti da ricercare.
    - Per richieste su funzionalità standard, entrambe le ricerche (semantica e keyword) possono essere utili.
    - Per richieste più generiche su plugin o personalizzazioni, la ricerca semantica aiuta a trovare anche concetti correlati,
    mentre la ricerca per keyword permette di trovare riferimenti precisi ad aziende, clienti o tipologie di plugin nelle RI.

    I due sistemi di ricerca:
    1. Ricerca semantica: adatta per concetti generali, macro-argomenti o combinazioni di funzionalità.
    2. Ricerca per keyword: adatta per termini specifici, nomi di moduli, aziende o funzionalità precise, in particolare plugin/customizzazioni.

    Rispondi SEMPRE in formato JSON:
    {{
        "use_semantic": True/False,
        "use_keyword": True/False,
        "reason": "<spiegazione breve della decisione, indicando se la richiesta riguarda standard, plugin o entrambe le tipologie>"
    }}

    Esempi di risposta attesa:
    {few_shot_text}
    """

    user_message = f"Prompt utente: {prompt}\nDecisione:"

    response = client.chat.completions.create(
        model = MODEL_NAME,
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1,
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
def select_documents(user_prompt: str, chunks: list) -> dict:
    
    # STEP 3a
    # Rimozione duplicati segnalando importanza maggiore 
    seen = {}
    for doc in chunks:
        key = (doc["numero"], doc["progressivo"])
        if key not in seen:
            seen[key] = doc
        else:
            existing = seen[key].get("retrieval_sources", [])
            incoming = doc.get("retrieval_sources", [])
            seen[key]["retrieval_sources"] = list(set(existing + incoming))
            if doc.get("similarity", 0) > seen[key].get("similarity", 0):
                seen[key]["similarity"] = doc["similarity"]
    chunks = list(seen.values())

    # Carico template
    doc = Document("main/evaluation/Sample_RI.docx")
    template_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    # STEP 3b
    # Filtro i chunk simili al template base
    documents = []
    removed_indices = []
    similarity_threshold=0.9
    for i, chunk in enumerate(chunks):
        sim_template = SequenceMatcher(None, chunk['content'], template_text).ratio()
        if sim_template < similarity_threshold:
            documents.append(chunk)
        else:
            removed_indices.append(i)

    if not documents:
        # Se tutti i chunk sono simili al template, restituisco fallback
        return {
            "relevant_docs": [],
            "irrelevant_docs": list(range(len(chunks))),
            "reason": "Tutti i chunk sono simili al template di base e quindi non utili."
        }

    # STEP 4
    examples = [
        {
            "prompt": "Come posso generare un report dei movimenti di magazzino?",
            "documents": [
                {"titolo": "Report vendite mensili", "content": "Contiene dati di vendita per cliente e prodotto.", "autore": "Mario", "cliente": "ClienteA", "similarity": 0.42},
                {"titolo": "Movimenti di magazzino", "content": "Elenco dettagliato dei movimenti di magazzino con filtri per data e prodotto nel modulo standard di SAM ERP2.", "autore": "Luigi", "cliente": "ClienteB", "similarity": 0.91},
                {"titolo": "Guida configurazione magazzino", "content": "Istruzioni su come configurare il modulo magazzino nel software, utile per eventuali personalizzazioni dei report.", "autore": "Anna", "cliente": "ClienteC", "similarity": 0.75}
            ],
            "decision": {
                "relevant_docs": [1, 2],
                "irrelevant_docs": [0],
                "reason": "Il secondo documento contiene informazioni dirette sui movimenti di magazzino standard. Il terzo è utile come supporto tecnico per configurazioni o personalizzazioni. Il primo documento non è rilevante per la richiesta."
            }
        },
        {
            "prompt": "Come posso impostare le chiavi univoche per i clienti?",
            "documents": [
                {"titolo": "Chiavi univoche clienti", "content": "Spiega come configurare chiavi univoche per evitare duplicazioni di dati cliente nel modulo anagrafica clienti standard.", "autore": "Luca", "cliente": "ClienteX", "similarity": 0.95},
                {"titolo": "Gestione magazzini", "content": "Istruzioni per creare nuovi magazzini e assegnare codici identificativi.", "autore": "Marco", "cliente": "ClienteY", "similarity": 0.55},
                {"titolo": "Parametri generali azienda", "content": "Descrive le impostazioni generali dell'azienda, ma non tratta le chiavi univoche.", "autore": "Elisa", "cliente": "ClienteZ", "similarity": 0.40}
            ],
            "decision": {
                "relevant_docs": [0],
                "irrelevant_docs": [1, 2],
                "reason": "Solo il primo documento tratta direttamente la funzionalità standard di chiavi univoche. Gli altri documenti non sono pertinenti."
            }
        },
        {
            "prompt": "Perché ricevo l'errore 'magazzino non trovato' quando salvo un ordine?",
            "documents": [
                {"titolo": "Errori comuni nella gestione ordini", "content": "L'errore 'magazzino non trovato' si verifica quando il magazzino indicato non è attivo o non è associato all'azienda selezionata.", "autore": "Verdi", "cliente": "Cliente1", "similarity": 0.92},
                {"titolo": "Configurazione magazzino", "content": "Descrive come attivare un magazzino e associarlo a un'azienda dal menu Anagrafica Magazzini standard e plugin opzionali.", "autore": "Rossi", "cliente": "Cliente2", "similarity": 0.88},
                {"titolo": "Report vendite annuali", "content": "Analisi delle vendite su base annuale con filtri per categoria e zona geografica.", "autore": "Neri", "cliente": "Cliente3", "similarity": 0.33}
            ],
            "decision": {
                "relevant_docs": [0, 1],
                "irrelevant_docs": [2],
                "reason": "I primi due documenti contengono informazioni dirette sul problema e sulla risoluzione, sia in funzionalità standard che in plugin. Il terzo documento non è rilevante."
            }
        },
        {
            "prompt": "Come posso modificare la fattura di un cliente?",
            "documents": [
                {"titolo": "Fatture clienti", "content": "Guida alla gestione delle fatture e modifica dei dati cliente nel modulo contabilità standard.", "autore": "Andrea", "cliente": "ClienteD", "similarity": 0.60},
                {"titolo": "Gestione magazzino", "content": "Movimenti di magazzino e scorte.", "autore": "Riccardo", "cliente": "ClienteE", "similarity": 0.42}
            ],
            "decision": {
                "relevant_docs": [0],
                "irrelevant_docs": [1],
                "reason": "Solo il primo documento tratta la modifica delle fatture (funzionalità standard). Il secondo non è pertinente."
            }
        }
    ]

    # Costruisco il few-shot text
    few_shot_text = ""
    for ex in examples:
        docs_text = "\n".join([f"{i}: {d['titolo']} - {d['content']} (autore: {d['autore']}, cliente: {d['cliente']})" for i, d in enumerate(ex['documents'])])
        few_shot_text += f"Prompt utente: {ex['prompt']}\nDocumenti:\n{docs_text}\nDecisione: {json.dumps(ex['decision'])}\n\n"

    system_message = f"""
    Sei un assistente decisionale incaricato di filtrare e riordinare per rilevanza i documenti estratti da un sistema 
    di ricerca per il software gestionale SAM ERP2 dell'azienda Centro Software. 
    Devi identificare quali documenti tra quelli estratti sono strettamente utili per rispondere a una richiesta utente e coerenti
    con essa nello specifico, tenendo sempre conto delle funzionalità standard e di quelle offerte solo tramite plugin/customizzazioni. 
    Durante la classificazione e la creazione delle liste di indici, riordina gli indici dei documenti utili in ordine crescente di rilevanza, dal meno al più rilevante.

    Nota:
    - La ricerca documentale restituisce sempre i top 25 documenti più rilevanti, ma non tutti sono effettivamente utili.
    - Seleziona solo i documenti più coerenti con la richiesta dell'utente, fino ad un massimo di 15.
    - Per ciascun documento è disponibile il contenuto e il valore di similarità coseno con il prompt utente.
    Usa queste informazioni come guida, ma considera anche il contenuto reale in termini di utilità per la risposta e la coerenza con la domanda.
    - Alcuni chunk potrebbero essere identici al template di base e quindi privi di utilità.
    - Se un documento riporta retrieval_sources: ["semantic", "keyword"], significa che è stato recuperato da entrambe le tecniche di ricerca: questo è un segnale forte di rilevanza e va considerato prioritario nella selezione.

    Criteri di rilevanza:
    1. Un documento è rilevante se contiene informazioni concrete che aiutano a rispondere 
    alla richiesta, sia su funzionalità standard che su plugin/customizzazioni.
    2. Alcuni documenti possono essere solo moderatamente coerenti o riferirsi a plugin 
    non pertinenti alla richiesta: questi vanno considerati non utili.
    3. Considera la differenza tra standard e plugin: se il prompt riguarda una funzionalità standard, anche documenti 
    sui plugin correlati possono essere utili, mentre se riguarda plugin specifici, concentrati sui documenti che li contengono.

    Rispondi SEMPRE in formato JSON:
    {{
        "relevant_docs": [indici dei documenti utili in ordine crescente di rilevanza],
        "irrelevant_docs": [indici dei documenti non utili],
        "reason": "<breve spiegazione della decisione, indicando se la rilevanza dipende da standard, plugin o entrambe le tipologie>"
    }}

    Esempi di risposta attesa:
    {few_shot_text}
    """

    # Prompt utente per il modello con i chunk reali
    docs_text = "\n".join([
        f"{i}: {d['titolo']} - {d['content']} "
        f"(autore: {d['autore']}, cliente: {d['cliente']}, "
        f"sorgenti: {'+'.join(d.get('retrieval_sources', ['unknown']))})"
        for i, d in enumerate(documents)
    ])
    user_message = f"{few_shot_text}Prompt utente: {user_prompt}\nDocumenti:\n{docs_text}\nDecisione:"

    # Chiamata al modello
    response = client.chat.completions.create(
        model = MODEL_NAME,
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1,
    )

    decision_text = response.choices[0].message.content

    # Parsing JSON con fallback
    try:
        decision_filtered = json.loads(decision_text)
        # Ricostruiamo gli indici rispetto ai chunk originali
        relevant_docs = [documents[i] for i in decision_filtered.get("relevant_docs", [])]
        irrelevant_docs = removed_indices + [documents[i] for i in decision_filtered.get("irrelevant_docs", [])]
        decision = {
            "relevant_docs": [chunks.index(chunk) for chunk in relevant_docs],
            "irrelevant_docs": [chunks.index(chunk) if isinstance(chunk, dict) else chunk for chunk in irrelevant_docs],
            "reason": decision_filtered.get("reason", "Decisione generata dall'LLM")
        }
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
        f"Numero RI: {d['numero']} - Titolo: {d['titolo']}, Chunk: {d['progressivo']}, "
        f"Autore: {d['autore']}, Cliente: {d['cliente']}, "
        f"Sorgenti retrieval: {'+'.join(d.get('retrieval_sources', ['unknown']))}\n"
        f" - Contenuto: {d['content']}"
        for d in selected_docs
    ]) if selected_docs else "Nessun documento rilevante trovato."

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
            "answer": "Per creare un nuovo ordine cliente in **SAM ERP2**, accedi al modulo **Vendite → Ordini → Nuovo**. Compila i campi obbligatori (Cliente, Data, Magazzino e Condizioni di pagamento) e salva per confermare [1]. Assicurati che il cliente sia già presente in anagrafica [2].\n\nDocumenti di riferimento:\n1. RI: [41699](https://intranet.centrosoftware.com/IntraCSW/script/vedi_RI.asp?idRI=41699) - Creazione Ordine Cliente, Chunk: 3, Autore: Rossi, Cliente: Centro Software\n2. RI: [41700](https://intranet.centrosoftware.com/IntraCSW/script/vedi_RI.asp?idRI=41700) - Gestione anagrafiche clienti, Chunk: 1, Autore: Bianchi, Cliente: Centro Software"
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
            "answer": "L'errore **'magazzino non trovato'** in **SAM ERP2** indica che il magazzino selezionato non è attivo o non è associato all'azienda corrente [1]. Per risolvere, vai in **Magazzini → Anagrafica**, verifica che il magazzino sia attivo (flag 'Attivo' selezionato) e collegato all'azienda corretta [2].\n\nDocumenti di riferimento:\n1. RI: [42210](https://intranet.centrosoftware.com/IntraCSW/script/vedi_RI.asp?idRI=42210) - Errori comuni nella gestione ordini, Chunk: 7, Autore: Verdi, Cliente: Centro Software\n2. RI: [42211](https://intranet.centrosoftware.com/IntraCSW/script/vedi_RI.asp?idRI=42211) - Configurazione magazzini, Chunk: 2, Autore: Neri, Cliente: Centro Software"
        },
        {
            "user_prompt": "Qual è la politica aziendale per le ferie dei dipendenti?",
            "documents": [],
            "answer": "Mi dispiace, ma non ho trovato informazioni sulle politiche ferie nei documenti disponibili. Posso però aiutarti a indirizzare la richiesta al reparto Risorse Umane o fornire indicazioni generali se mi dai maggiori dettagli."
        },
        {
            "user_prompt": "Come configurare i piani di consegna e le chiavi univoche?",
            "documents": [
                {
                    "numero": "12121",
                    "titolo": "Configurazione piani di consegna",
                    "progressivo": "4",
                    "content": "Per impostare i piani di consegna, definire importazione e gestione dei dati nel modulo standard di SAM ERP2.",
                    "autore": "Mario",
                    "cliente": "Alfa"
                },
                {
                    "numero": "12345",
                    "titolo": "Gestione chiavi univoche",
                    "progressivo": "0",
                    "content": "Le chiavi univoche devono essere definite per ogni cliente per evitare conflitti, anche in personalizzazioni e plugin su richiesta del cliente.",
                    "autore": "Luca",
                    "cliente": "Beta"
                }
            ],
            "answer": "Per configurare i piani di consegna, definisci importazione e gestione dei dati nel modulo standard di **SAM ERP2** [1]. Le chiavi univoche devono essere definite per ciascun cliente, anche per eventuali plugin o personalizzazioni [2].\n\nDocumenti di riferimento:\n1. RI: [12121](https://intranet.centrosoftware.com/IntraCSW/script/vedi_RI.asp?idRI=12121) - Configurazione piani di consegna, Chunk: 5, Autore: Mario, Cliente: Alfa\n2. RI: [12345](https://intranet.centrosoftware.com/IntraCSW/script/vedi_RI.asp?idRI=12345) - Gestione chiavi univoche, Chunk: 1, Autore: Luca, Cliente: Beta"
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
                    "cliente": "Gamma"
                }
            ],
            "answer": "Per verificare i dati importati nel modulo standard di **SAM ERP2**, controlla che tutti i campi siano completi e corretti [1].\n\nDocumenti di riferimento:\n1. RI: [51000](https://intranet.centrosoftware.com/IntraCSW/script/vedi_RI.asp?idRI=51000) - Controllo dati importati, Chunk: 4, Autore: Anna, Cliente: Gamma"
        },
        {
            "user_prompt": "L'email per contattare GMR ENLIGHTS S.R.L. è info@gmrenlights.it, è corretta?",
            "documents": [
                {
                    "numero": "55002",
                    "titolo": "Anagrafica GMR ENLIGHTS",
                    "progressivo": "0",
                    "content": "Cliente: GMR ENLIGHTS S.R.L. — Email: info@gmrenlights.com — Referente: Daniele Valentini.",
                    "autore": "Rossi",
                    "cliente": "GMR ENLIGHTS S.R.L."
                }
            ],
            "answer": "L'email indicata nella tua domanda non è corretta. Secondo i documenti disponibili, "
                    "l'indirizzo email di **GMR ENLIGHTS S.R.L.** è **info@gmrenlights.com** (.com, non .it) [1].\n\n"
                    "Documenti di riferimento:\n"
                    "1. RI: [55002](https://intranet.centrosoftware.com/IntraCSW/script/vedi_RI.asp?idRI=55002) "
                    "- Anagrafica GMR ENLIGHTS, Chunk: 1, Autore: Rossi, Cliente: GMR ENLIGHTS S.R.L."
        }
    ]

    few_shot_text = ""
    for ex in examples:
        docs_text_example = "\n".join([f"{i+1}: Numero RI: {d['numero']} - Titolo: {d['titolo']}, Autore: {d['autore']}, Cliente: {d['cliente']}\n - Contenuto: {d['content']}" for i, d in enumerate(ex['documents'])
        ])
        few_shot_text += f"Prompt utente: {ex['user_prompt']}\nDocumenti:\n{docs_text_example}\nRisposta attesa:\n{ex['answer']}\n\n"

    system_message = f"""
    Sei un assistente esperto del software gestionale SAM ERP2 dell'azienda Centro Software. 
    Conosci le funzionalità standard del software e la distinzione tra funzioni native e plugin/customizzazioni sviluppate su richiesta dei clienti. 
    Devi rispondere alle richieste degli utenti usando solo le informazioni presenti nei documenti forniti, integrando le tue conoscenze pregresse solo per distinguere cosa è standard e cosa è personalizzato. 

    Regole:
    1. Non inventare informazioni o dettagli che non sono nei documenti forniti. 
    2. Se una funzionalità è standard e nota dal tuo modello, puoi chiarirlo, ma evidenzia sempre quando una funzionalità è invece un plugin o una personalizzazione basata su Richieste di implementazione (RI) fornite nei documenti.
    3. Annotare ogni riferimento a un documento con un numero tra parentesi quadre [1], [2], ecc.
    4. Alla fine della risposta, fornire la lista dei documenti di riferimento usati. In questo formato ('numero'=41699 nell'esempio): 1. RI: [41699](https://intranet.centrosoftware.com/IntraCSW/script/vedi_RI.asp?idRI=41699) - 'titolo', Chunk: 'progressivo+1', Autore: 'autore', Cliente 'cliente'.
    5. Mantieni la risposta chiara, strutturata e basata esclusivamente sui documenti forniti per quanto riguarda le personalizzazioni.
    6. Se non vengono forniti documenti, rispondi in modo naturale e professionale, facendo riferimento solo alle funzionalità standard conosciute di SAM ERP2.
    7. Genera l'output in formato markdown, così che possa essere utilizzato direttamente all'interno di un'interfaccia Streamlit.
    8. Se le informazioni fornite dall'utente nella query contraddicono quanto riportato nei documenti (ad esempio un'email, un nome cliente o un parametro errato), segnala esplicitamente la discrepanza e riporta il dato corretto trovato nei documenti, senza confermare il dato errato dell'utente.
    
    Esempi di risposta attesa:
    {few_shot_text}
    """

    summarized_chat = summarize_chat_history(chat_history)

    user_message = f"""
    Prompt utente:
    {user_prompt}

    Contesto conversazionale recente:
    {summarized_chat}

    Documenti disponibili:
    {context_text}

    Risposta attesa:
    """

    response_stream = client.chat.completions.create(
        model = MODEL_NAME,
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.5,
        stream=True,
    )

    # Lettura dei token generati
    final_text = ""
    for event in response_stream:
        if not event.choices:
            continue

        choice = event.choices[0]
        delta = getattr(choice, "delta", None)
        if delta and getattr(delta, "content", None):
            token = delta.content
            final_text += token
            yield token  # invia il token man mano che arriva
            

# FLUSSO COMPLETO DEL CHATBOT =====================================================
def gpt_request(messages):

    # Estrae l'ultimo prompt inserito dalla cronologia chat
    user_prompt = [m["content"] for m in messages if m["role"] == "user"][-1]

    # Decisione strumenti di retrieval
    tools = decide_tools(user_prompt)

    # STEP 2 - Recupero documenti
    all_documents = []
    if tools["use_semantic"]:
        print("\nUso la ricerca SEMANTICA\n")
        all_documents += semantic_search(user_prompt)

    if tools["use_keyword"]:
        print("\nUso la ricerca per KEYWORDS\n")
        all_documents += keyword_search(user_prompt)

    # 2 - Selezione documenti in base alla coerenza
    document_selection = []
    selected_docs = []
    if tools["use_semantic"] or tools["use_keyword"]:
        if all_documents:
            document_selection = select_documents(user_prompt, all_documents)
            print("\n\n===================================\n\n")
            selected_docs = [all_documents[i] for i in document_selection['relevant_docs']]

    # 3 - Genera risposta finale in modalità stream
    return generate_final_answer(user_prompt, selected_docs, messages)


# GESTIONE LUNGHEZZA CRONOLOGIA MESSAGGI =====================================================
def summarize_chat_history(chat_history, model_name=MODEL_NAME):

    if not chat_history:
        return ""
    
    encoding = tiktoken.encoding_for_model(model_name)
    max_tokens=10000
    reserved_tokens_for_summary = 2000

    # Se la chat è più corta del limite, restituisci tutta la chat senza ulteriori elaborazioni
    token_counts = [len(encoding.encode(f"{'Utente' if m['role']=='user' else 'Assistente'}: {m['content']}\n")) for m in chat_history]
    total_tokens = sum(token_counts)

    if total_tokens <= max_tokens:
        return "".join(f"{'Utente' if m['role']=='user' else 'Assistente'}: {m['content']}\n" for m in chat_history)
    
    recent_lines = []
    old_lines = []
    recent_tokens = 0
    old_tokens = 0

    # Salvo messaggi più vecchi (inizio)
    for m, t in zip(chat_history, token_counts):
        if old_tokens + t <= (max_tokens - reserved_tokens_for_summary) // 2:
            role = "Utente" if m["role"] == "user" else "Assistente"
            old_lines.append(f"{role}: {m['content']}\n")
            old_tokens += t
        else:
            break

    # Salvo messaggi più recenti (fine)
    for m, t in zip(reversed(chat_history), reversed(token_counts)):
        if recent_tokens + t <= (max_tokens - reserved_tokens_for_summary) // 2:
            role = "Utente" if m["role"] == "user" else "Assistente"
            recent_lines.insert(0, f"{role}: {m['content']}\n")  # inserisci in cima
            recent_tokens += t
        else:
            break
    
    # Riassumo messaggi centrali
    start_idx = len(old_lines)
    end_idx = len(chat_history) - len(recent_lines)
    central_messages = chat_history[start_idx:end_idx]
    summary_text = ""
    if central_messages:
        summary_text = summarize_old_messages(central_messages, reserved_tokens_for_summary)
        summary_text = f"Riassunto messaggi centrali: {summary_text}\n"

    # Unisci tutto
    return "".join(old_lines) + summary_text + "".join(recent_lines)

# RIASSUNTO MESSAGGI VECCHI
def summarize_old_messages(messages, max_tokens):

    model_name=MODEL_NAME

    if not messages:
        return ""
    
    system_prompt = """
    Sei un assistente che deve riassumere i messaggi della chat relativi al software gestionale SAM ERP2. 
    Il riassunto deve essere chiaro, breve e utile al contesto, mantenendo solo i concetti principali. 

    Criteri:
    1. Mantieni nel riassunto le informazioni rilevanti per capire il contesto delle richieste, indicando chiaramente se riguardano funzionalità standard o plugin.
    2. Sintetizza i concetti, rimuovendo dettagli ridondanti o non essenziali.
    3. Il riassunto deve fornire un contesto sufficiente per permettere all'LLM di rispondere correttamente alle richieste successive.
    """

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
        temperature=0.35,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content