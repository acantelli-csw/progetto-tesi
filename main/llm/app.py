import streamlit as st
import time
import ui
import llm

# Configurazine base
st.set_page_config(page_title="Assistente Documentale RI", page_icon="https://www.digitalrecruitingweek.it/wp-content/uploads/2023/03/CENTRO-SOFTWARE-logo.png", layout="centered")

# Applica stile
ui.apply_style()

# Inizializza stato sidebar
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

# Toggle apertura/chiusura
top_col1, top_col2 = st.columns([1, 4])
with top_col1:
    if st.button("☰"):
        st.session_state.sidebar_open = not st.session_state.sidebar_open

if st.session_state.sidebar_open:
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                min-width: 25rem !important;
                max-width: 42rem !important;
                transition: all 0.3s ease;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                width: 0 !important;
                min-width: 0 !important;
                max-width: 0 !important;
                overflow: hidden !important;
                transition: all 0.3s ease;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Costruisci barra laterale per accesso utente
with st.sidebar:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image("https://www.digitalrecruitingweek.it/wp-content/uploads/2023/03/CENTRO-SOFTWARE-logo.png", width=225)
    st.markdown("### 📄 Assistente Documentale RI")
    st.markdown("Accedi al tuo account e inizia a chattare con l'Assistente Documentale per ricerca RAG delle RI")
    st.markdown("---")

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

# Inizializza chat al primo avvio o recupera la cronologia chat al refresh della pagina
if "messages" not in st.session_state:
    st.session_state.messages = ui.load_chat_history(username)
    if not st.session_state.messages:
        st.session_state.messages = [
            {"role": "system", "content": "Sei un assistente documentale esperto del software gestionale BPM. Rispondi in modo chiaro e basandoti solo sui documenti o sulle informazioni già fornite."},
            {"role": "assistant", "content": "Come posso esserti utile? 👇"}]

# Testata principale chat
ui.load_testata()

# Mostra la cronologia messaggi della chat al ri-caricamento
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "📄"):
        st.markdown(message["content"])

# Input utente
if prompt := st.chat_input("Scrivi qui..."):
    # Salva e mostra messaggio utente
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"): 
        st.markdown(prompt)

    # Mostra la risposta dell'assistente in chat
    with st.chat_message("assistant", avatar="📄"):
        full_response = ""
        message_placeholder = st.empty()

        for token in llm.gpt_request(st.session_state.messages):
            full_response += token
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)

        # Aggiungi la risposta dell'assistente alla cronologia chat
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        ui.save_chat_history(username, st.session_state.messages)