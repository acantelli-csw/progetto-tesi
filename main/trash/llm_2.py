import openai
from .env import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT

load_dotenv(dotenv_path=".env")

openai.api_type = "azure"
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2023-05-15"

def ask_openai_function_call(user_prompt, tools=None, temperature=0.2, max_tokens=1000):
    """
    Chiamata LLM con supporto function-calling.
    Il modello può decidere autonomamente se chiamare uno dei tool definiti.
    """
    messages = [
        {"role": "system", "content": open("prompt.txt", "r", encoding="utf-8").read()},
        {"role": "user", "content": user_prompt}
    ]

    response = openai.ChatCompletion.create(
        engine=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        functions=tools or [],
        function_call="auto"  # il modello decide se chiamare funzioni
    )

    message = response['choices'][0]['message']

    # Restituiamo funzione chiamata (se presente) o contenuto testuale
    result = {}
    if "function_call" in message:
        result["function_call"] = message["function_call"]
    else:
        result["content"] = message["content"]

    return result
