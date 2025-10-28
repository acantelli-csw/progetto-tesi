import llm
import search

prompt = "cnfigurazione mrl"
decision = llm.decide_tools(prompt)
print("1. Risposta del chatbot:\n", decision)


prompt = "Come implementare un mrp"
decision = llm.decide_tools(prompt)
print("\n2. Risposta del chatbot:\n", decision)

