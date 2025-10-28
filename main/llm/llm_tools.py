import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

# ========================================
# CONFIGURAZIONE
# ========================================

# Carica variabili d'ambiente
load_dotenv()

# Inizializza il client Azure OpenAI
client = AzureOpenAI(
    azure_endpoint=os.getenv("LLM_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("LLM_VERSION")
)
MODEL_NAME = os.getenv("LLM_MODEL")

# Parametri del chatbot
MAX_HISTORY_LENGTH = 20
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1500

# Lettura del prompt di sistema da file
prompt_path = Path(__file__).parent / "prompts" / "prompt_1a.txt"
SYSTEM_PROMPT = prompt_path.read_text(encoding="utf-8")

# ========================================
# DEFINIZIONE TOOLS
# ========================================


# Definizione dei tool disponibili per il LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Cerca documenti rilevanti usando ricerca semantica basata sul significato. Utile per domande concettuali o quando serve comprendere il contesto.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "La query di ricerca"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Numero di documenti da recuperare",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "keyword_search",
            "description": "Cerca documenti usando parole chiave esatte. Utile per trovare termini specifici, nomi, codici, numeri di riferimento.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "Parole chiave da cercare"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Numero di documenti da recuperare",
                        "default": 5
                    }
                },
                "required": ["keywords"]
            }
        }
    }
]

"""Cerca documenti usando ricerca semantica basata su embeddings. 
                Ideale per: domande concettuali, ricerca per significato, quando il contesto è importante.
                Esempi: 'Come funziona X?', 'Qual è la procedura per Y?', 'Spiegami il concetto di Z'"""

"""Cerca documenti usando parole chiave esatte (full-text search).
                Ideale per: termini specifici, nomi propri, codici, numeri, date, riferimenti precisi.
                Esempi: 'Trova documenti con codice ABC123', 'Cerca Mario Rossi'"""

def decide_tools(prompt):

    function_schema = [
        {
            "name": "choose_tools",
            "description": "Decide which search tools to use based on user prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "use_semantic_search": {"type": "boolean", "description": "Use semantic search?"},
                    "use_keyword_search": {"type": "boolean", "description": "Use keyword search?"}
                },
                "required": ["use_semantic_search", "use_keyword_search"]
            }
        }
    ]

    response = client.chat_completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Decidi se usare gli strumenti di ricerca e quali, in base al prompt dell'utente."},
            {"role": "user", "content": prompt}
        ],
        functions=function_schema,
        function_call={"name": "choose_tools"}
    )

    # Parsing della risposta
    function_response = response.choices[0].message.get("function_call")
    if function_response and "arguments" in function_response:
        return json.loads(function_response["arguments"])
    else:
        # Default: usare entrambi
        return {"use_semantic_search": True, "use_keyword_search": True}
    

#
# # ========================================
# GENERAZIONE RISPOSTA FINALE
# # ========================================

def generate_final_response(user_prompt, documents):

    context_text = "\n\n".join(documents) if documents else "Nessun documento rilevante trovato."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Domanda: {user_prompt}\n\nDocumenti:\n{context_text}"}
    ]

    response = client.chat_completions.create(
        model=MODEL_NAME,
        messages=messages
    )

    return response.choices[0].message["content"]


# ---------------------------------------------------
# FLUSSO COMPLETO DEL CHATBOT
# ---------------------------------------------------
def handle_user_prompt(user_prompt):
    # 1️. Decidi strumenti
    tools = decide_tools(user_prompt)
    
    # 2️. Recupero documenti
    documents = []
    if tools.get("use_semantic_search"):
        documents += search.semantic_search(user_prompt)
    if tools.get("use_keyword_search"):
        documents += keyword_search(user_prompt)

    # Rimuovi duplicati
    documents = list(dict.fromkeys(documents))

    # 3️⃣ Genera risposta finale
    final_response = generate_final_response(user_prompt, documents)
    return final_response

# ---------------------------------------------------
# ESEMPIO DI UTILIZZO
# ---------------------------------------------------
if __name__ == "__main__":
    prompt = "Spiegami come funziona il meccanismo di tokenizzazione in NLP."
    risposta = handle_user_prompt(prompt)
    print("Risposta del chatbot:\n", risposta)



"""
# Chiamata al modello con tools
response = client.chat.completions.create(
    model=deployment,
    messages=messages,
    functions=tools,
    temperature=temperature,
    max_tokens=max_tokens
    #functions=tools or [],
    #function_call="auto"  # il modello decide se chiamare funzioni
)

"""
