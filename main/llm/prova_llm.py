import llm
import time

#prompt = "configurazione piani di consegna"
#decision = llm.gpt_request(prompt)
#print(repr(decision))

def prova_chatbot (prompt):
    time.sleep(3)
    print("Eseguendo chiamata LLM")
    return(f"Risposta di prova al prompt: {prompt}")
