@ -1,68 +0,0 @@
from utils.azure_openai import ask_openai_function_call
from tools.semantic_search import semantic_search
from tools.keyword_search import keyword_search

def main():
    user_query = input("Inserisci la tua richiesta: ")

    # Definizione dei tool disponibili per il LLM
    tools = [
        {
            "name": "semantic_search",
            "description": "Esegue una ricerca semantica sui documenti e restituisce i chunk più rilevanti.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query utente"},
                    "top_k": {"type": "integer", "description": "Numero massimo di chunk da restituire", "default": 20}
                },
                "required": ["query"]
            }
        },
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
    ]

    # LLM decide se chiamare uno o entrambi i tools
    response = ask_openai_function_call(
        user_prompt=user_query,
        tools=tools
    )

    # Interpretiamo eventuale chiamata a funzione richiesta dal modello
    if response.get("function_call"):
        function_name = response["function_call"]["name"]
        args = response["function_call"]["arguments"]

        if function_name == "semantic_search":
            results = semantic_search(args["query"], top_k=args.get("top_k", 20))
        elif function_name == "keyword_search":
            results = keyword_search(args["query"])
        else:
            results = []

        # Invio al modello i risultati per la risposta finale
        final_response = ask_openai_function_call(
            user_prompt=f"Questi sono i risultati ottenuti dal tool {function_name}:\n{results}\nGenera la risposta finale all'utente usando questi dati.",
            tools=[]  # nessun tool necessario in questo step
        )

        print("\nRISPOSTA FINALE DEL CHATBOT:\n")
        print(final_response["content"])
    else:
        # Se l'LLM non ha chiamato funzioni, restituisce direttamente la risposta
        print("\nRISPOSTA FINALE DEL CHATBOT:\n")
        print(response["content"])


if __name__ == "__main__":
    main()