import streamlit as st
import time
import re
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

    st.markdown("### 🔐 Accesso Utente")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None

    # Se l'utente non è autenticato → mostra login/registrazione
    if not st.session_state.authenticated:
        action = st.radio("Seleziona azione:", ["Accedi", "Registrati"], horizontal=True)
    
        with st.form(key="auth_form", clear_on_submit=False):
            username_input = st.text_input("👤 Nome utente", key="username_input", label_visibility="visible")
            password_input = st.text_input("🔑 Password", key="password_input", label_visibility="visible")
            submit = st.form_submit_button("Login" if action == "Accedi" else "Crea account")

        if action == "Accedi" and submit:
            if ui.authenticate_user(username_input, password_input):
                st.session_state.authenticated = True
                st.session_state.username = username_input

                # Carica subito la chat dell'utente
                st.session_state.messages = ui.load_chat_history(username_input)
                if not st.session_state.messages:
                    st.session_state.messages = [{"role": "assistant", "content": "Come posso esserti utile? 👇"}]

                st.success(f"Benvenuto, **{username_input}** 👋")
                st.rerun()
            else:
                st.error("Nome utente o password errati.")

        elif action == "Registrati" and submit:
            if not username_input or not password_input:
                st.warning("Inserisci sia username che password.")
            elif ui.register_user(username_input, password_input):
                st.success("Registrazione completata! Ora puoi accedere.")
            else:
                st.error("Questo nome utente esiste già.")
        st.stop()  # blocca il resto dell'app finché non è autenticato

    # Se autenticato → mostra opzioni utente
    else:
        username = st.session_state.username
        st.success(f"✅ Autenticato come **{username}**")

        # Pulsante reset chat
        if st.button("🔄 Resetta chat"):
            ui.reset_chat_history(username)
            st.session_state.messages = [{"role": "assistant", "content": "Chat resettata. Come posso aiutarti? 👇"}]
            ui.save_chat_history(username, st.session_state.messages)
            st.success("💬 Chat resettata con successo!")
            st.rerun()

        # Pulsante logout
        if st.button("🚪 Esci"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.messages = []
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
            # assistant_response = prova_llm.prova_chatbot(prompt)
            assistant_response = llm.gpt_request(prompt)

        # Simulate stream of response with milliseconds delay
        message_placeholder = st.empty()
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.03)

            # Pulizia e formattazione per Markdown
            formatted_response = full_response
            formatted_response = formatted_response.replace("\n", "  \n")
            formatted_response = re.sub(r"(?<!\n)\s*([-*]|\d+\.) ", r"\n\1 ", formatted_response)

            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(formatted_response + "▌", unsafe_allow_html=True)
        
        formatted_response = re.sub(r"(?<!\n)\s*([-*]|\d+\.) ", r"\n\1 ", full_response.replace("\n", "  \n"))
        message_placeholder.markdown(formatted_response, unsafe_allow_html=True)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": formatted_response})
    ui.save_chat_history(username, st.session_state.messages)