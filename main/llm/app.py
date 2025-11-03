import streamlit as st
import time
import llm
import ui 

# TODO: add cronologia chat precedenti nella sidebar laterale
# TODO: deciedere logo o icone da mettere (es. logo CSW/BPM/Costum) al psoto dell'emoji

# Configurazine base
st.set_page_config(page_title="Assistente Documentale RI", page_icon="📄", layout="wide")

# Applica stile personalizzato
ui.apply_style()

# Costruisci barra laterale
ui.build_sidebar()

# Inizializza chat al primo avvio
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Come posso esserti utile? 👇"}]

# Testata principale
st.markdown("<h2 style='text-align:center;'>📄 Assistente Documentale RI</h2>", unsafe_allow_html=True)
st.write("Benvenuto nel chatbot di BPM per la ricerca RAG delle RI!")
st.write("Posso aiutarti a trovare facilmente le informazioni che ti servono all'interno delle RI già sviluppate dai tuoi colleghi — così non dovrai ricominciare da zero!")

st.caption("Mi raccomando, verifica sempre i risultati ottenuti! Posso sbagliare anche io... \t Per farlo puoi controllare direttamente le RI utilizzate per generare la risposta e fornite con essa.")
st.divider()

# Mostra la cronologia messaggi della chat al ri-caricamento
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "📄"):
        st.markdown(message["content"])

# Input utente
if prompt := st.chat_input("Come implementare un piano di consegna..."):
    # Salva e mostra messaggio utente
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"): 
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="📄"):
        full_response = ""
        assistant_response = llm.gpt_request(prompt)

        # Simulate stream of response with milliseconds delay
        message_placeholder = st.empty()
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            markdown_ready = full_response.replace("\n- ", "\n- ")
            formatted_response = markdown_ready.replace("\n", "  \n")
            message_placeholder.markdown(formatted_response + "▌")
        message_placeholder.markdown(formatted_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": formatted_response})