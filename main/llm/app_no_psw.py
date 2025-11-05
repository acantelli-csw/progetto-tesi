import streamlit as st
import time
import ui
import llm
import prova_llm

# TODO: add cronologia chat precedenti nella sidebar laterale
# TODO: deciedere logo o icone da mettere (es. logo CSW/BPM/Costum) al psoto dell'emoji

# Configurazine base
st.set_page_config(page_title="Assistente Documentale RI", page_icon="📄", layout="wide")

# Applica stile personalizzato
ui.apply_style()

# Costruisci barra laterale per accesso utente
with st.sidebar:
    st.image("https://www.digitalrecruitingweek.it/wp-content/uploads/2023/03/CENTRO-SOFTWARE-logo.png", width=120)
    st.markdown("### 📄 Assistente Documentale RI")
    st.markdown("Accedi al tuo account e chatta con il motore RAG aziendale per ricerca delle RI")
    st.divider()

    st.markdown("### 👤 Accesso Utente")

    username = st.text_input("Inserisci il tuo nome utente per iniziare a chattare o per recuperare la chat precedente:", key="username")
    if not username:
        st.warning("Se non inserisci il nome utente, la tua chat verrà cancellata ad ogni riavvio.")
        st.stop()
    
    st.markdown("Per cambiare utente ricarica la pagina ed inserisci il nome del nuovo utente desiderato.")

    # Bottone per resettare la chat
    if st.button("🔄 Resetta chat"):
        ui.reset_chat_history(username)
        st.session_state.messages = [{"role": "assistant", "content": "Chat resettata. Come posso aiutarti? 👇"}]
        ui.save_chat_history(username, st.session_state.messages)
        st.success("Chat resettata con successo!")
        st.rerun()

# Inizializza chat al primo avvio o recupera la cronolgia chat al refresh della pagina
if "messages" not in st.session_state:
    st.session_state.messages = ui.load_chat_history(username)
    if not st.session_state.messages:
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

        with st.spinner(" Sto elaborando la risposta..."):
            assistant_response = prova_llm.prova_chatbot(prompt)
            # assistant_response = llm.gpt_request(prompt)

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
    ui.save_chat_history(username, st.session_state.messages)