import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
from search import semantic_search

# Carica variabili d'ambiente
load_dotenv()

# Inizializza il client Azure OpenAI
client = AzureOpenAI(
    azure_endpoint=os.getenv("LLM_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("LLM_VERSION")
)

# Lettura del prompt di sistema da file
prompt_path = Path(__file__).parent / "prompts" / "prompt_1a.txt"
system_prompt = prompt_path.read_text(encoding="utf-8")

# Definizione dei tool disponibili per il LLM
tools = [
    {
        "name": "semantic_search",
        "description": "Esegue una ricerca semantica sui documenti e restituisce i chunk più rilevanti.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Prompt utente"},
            },
            "required": ["prompt"]
        }
    }
]
""",
    {
        "name": "keyword_search",
        "description": "Esegue una ricerca testuale per keyword nei documenti.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query utente"}
            },
            "required": ["query"]
        }
    }
"""

def ask_gpt(user_prompt: str, temperature: float = 0.7, max_tokens: int = 500):

    deployment = os.getenv("LLM_MODEL")
    if not deployment:
        raise ValueError("Deployment non configurato")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

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

    choice = response.choices[0]
    
    message = response['choices'][0]['message']

    # Restituiamo funzione chiamata (se presente) o contenuto testuale
    result = {}
    if "function_call" in message:
        result["function_call"] = message["function_call"]
    else:
        result["content"] = message["content"]

    return result