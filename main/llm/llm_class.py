class RAGChatbot:
    def __init__(self, azure_endpoint, api_key, deployment_name):
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-01"
        )
        self.deployment = deployment_name
        self.conversation_history = []
        
    def chat(self, user_message):
        # Aggiungi messaggio utente alla history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Prima chiamata: LLM decide se usare tool
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=self.conversation_history,
            tools=tools,
            tool_choice="auto"  # LLM decide autonomamente
        )
        
        # Gestisci la risposta
        return self._handle_response(response)
    
    def _handle_response(self, response):
        message = response.choices[0].message
        
        # Caso 1: LLM vuole usare tool
        if message.tool_calls:
            return self._execute_tools(message)
        
        # Caso 2: Risposta diretta senza tool
        else:
            self.conversation_history.append({
                "role": "assistant",
                "content": message.content
            })
            return message.content
    
    def _execute_tools(self, assistant_message):
        # Salva la richiesta di tool nella history
        self.conversation_history.append(assistant_message)
        
        # Esegui ogni tool richiesto
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            # Esegui il tool appropriato
            if function_name == "semantic_search":
                result = self.semantic_search(**arguments)
            elif function_name == "keyword_search":
                result = self.keyword_search(**arguments)
            
            # Aggiungi risultato alla history
            self.conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
        
        # Seconda chiamata: LLM genera risposta con i risultati
        final_response = self.client.chat.completions.create(
            model=self.deployment,
            messages=self.conversation_history
        )
        
        final_message = final_response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": final_message
        })
        
        return final_message