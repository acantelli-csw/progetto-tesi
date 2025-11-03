import llm

prompt = "configurazione piani di consegna"
decision = llm.gpt_request(prompt)
print(repr(decision))
