import llm

prompt = "configurazione piani di consegna"
decision = llm.gpt_request(prompt)
print("\nRisposta del chatbot:\n", decision)
