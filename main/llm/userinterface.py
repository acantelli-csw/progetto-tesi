import streamlit as st
import time
import main.llm.main_llm as main_llm

st.write("Benvenuto nel chatbot di BPM!")
st.write("Posso aiutarti a trovare facilmente le informazioni che ti servono all’interno delle RI già sviluppate dai tuoi colleghi — così non dovrai ricominciare da zero!")

st.caption("Mi raccomando, verifica sempre i risultati ottenuti! Posso sbagliare anche io...")
st.caption("Per farlo puoi controllare direttamente le RI utilizzate per generare la risposta e fornite con essa.")

# Inizializza cronologia chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Come posso esserti utile? 👇"}]

# Mostra messaggi precedenti al ri-caricamento
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input utente
if prompt := st.chat_input("Come implementare un piano di consegna?"):
    # Salva e mostra messaggio utente
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = main_llm.llm_call(prompt)

        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})