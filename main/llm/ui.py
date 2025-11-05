import streamlit as st
import json
import os

def apply_style():
    """Applica CSS personalizzato per look aziendale sobrio e chat destra/sinistra."""
    st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }

    /* Contenitore messaggi */
    [data-testid="stChatMessage"] {
        border-radius: 0.75rem;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        display: inline-block;
        max-width: 70%;
        word-wrap: break-word;
        clear: both;
    }

    /* Messaggi utente a destra */
    [data-testid="stChatMessage"][data-testid="user"] {
        background-color: #e9ecef;
        float: right;
        text-align: right;
    }

    /* Messaggi assistant a sinistra */
    [data-testid="stChatMessage"][data-testid="assistant"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        float: left;
        text-align: left;
    }

    .stChatInputContainer {
        background-color: white !important;
        border-radius: 0.75rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dee2e6;
    }

    /* Clear float dopo la chat */
    [data-testid="stVerticalBlock"] {
        overflow: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
def build_sidebar():
    with st.sidebar:
        st.image("https://www.digitalrecruitingweek.it/wp-content/uploads/2023/03/CENTRO-SOFTWARE-logo.png", width=120)
        st.markdown("### 📄 Assistente Documentale RI")
        st.markdown("Chatta con il motore RAG aziendale per ricerca delle RI")
        st.divider()

        st.markdown("### 👤 Impostazioni Utente")
        username = st.text_input("Inserisci il tuo nome utente:", key="username")
        if not username:
            st.warning("Inserisci un nome utente per iniziare la chat.")
            st.stop()

        # Bottone per resettare la chat
        if st.button("🔄 Resetta chat"):
            reset_chat_history(username)
            st.session_state.messages = [{"role": "assistant", "content": "Chat resettata. Come posso aiutarti? 👇"}]
            save_chat_history(username, st.session_state.messages)
            st.success("Chat resettata con successo!")
            st.rerun()

def get_user_filename(username):
    return f"chat_history\chat_history_{username}.json"

def load_chat_history(username):
    filename = get_user_filename(username)
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_chat_history(username, messages):
    filename = get_user_filename(username)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def reset_chat_history(username):
    filename = get_user_filename(username)
    if os.path.exists(filename):
        os.remove(filename)
