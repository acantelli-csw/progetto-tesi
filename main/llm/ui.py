import streamlit as st
import json
import os
import hashlib

def apply_style():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        display: flex;
        flex-direction: column;
        height: 100vh;
    }

    /* Contenitore principale Streamlit (sidebar + contenuto) */
    [data-testid="stHorizontalBlock"] {
        display: flex;
        flex-wrap: nowrap;
        width: 100%;
        height: 100%;
        overflow: hidden;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dee2e6;
        width: 20%;
        min-width: 200px;
        max-width: 700px;
        transition: all 0.1s ease-in-out;
    }

    /* Contenuto principale */
    [data-testid="stVerticalBlock"]:not(section[data-testid="stSidebar"] *) {
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: stretch;
        padding: 1rem;
        overflow-y: auto;  
        box-sizing: border-box;
    }
        
    /* Contenitore messaggi chat */
    [data-testid="stChatMessage"] {
        border-radius: 0.75rem;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        max-width: 70%;
        word-wrap: break-word;
        display: block;
    }

    /* Messaggi utente (a destra) */
    [data-testid="stChatMessage"][data-testid="user"] {
        background-color: #e9ecef;
        margin-left: auto;
    }

    /* Messaggi assistant (a sinistra) */
    [data-testid="stChatMessage"][data-testid="assistant"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        margin-right: auto;
    }

    /* Campo input chat */
    .stChatInputContainer {
        background-color: white !important;
        border-radius: 0.75rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        width: 100%;
        max-width: 800px;
        margin-top: auto;
    }

    /* Hover su input base */
    div[data-baseweb="input"]:hover > div {
        background-color: #f8f9fa !important;
        border-radius: 0.75rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }

    /* Input nome utente */
    div[data-baseweb="input"] > div {
        background-color: #f8f9fa !important;
        border: 1.5px solid #dee2e6 !important;
        box-shadow: 0 0 6px rgba(0,123,255,0.25);
        border-radius: 0.5rem !important;
        padding: 0.4rem 0.8rem !important;
        transition: all 0.2s ease-in-out;
    }

    /* Responsività */
    @media (max-width: 768px) {
        [data-testid="stHorizontalBlock"] {
            flex-direction: column;
        }
        section[data-testid="stSidebar"] {
            width: 100%;
            max-width: none;
            border-right: none;
            border-bottom: 1px solid #dee2e6;
        }
        [data-testid="stVerticalBlock"]:not(section[data-testid="stSidebar"] *) {
            padding: 0.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def load_testata():
    st.markdown("# 📄 Assistente Documentale RI", unsafe_allow_html=True)
    st.markdown(" ")
    st.write("Benvenuto! Sono il tuo assistente per la ricerca RAG delle RI.")
    st.write("Posso aiutarti a trovare facilmente le informazioni che ti servono all'interno delle RI già sviluppate dai tuoi colleghi.")

    st.caption("Mi raccomando, verifica sempre i risultati ottenuti! Posso sbagliare anche io...")
    st.caption("Per farlo puoi controllare direttamente le RI utilizzate per generare la risposta e fornite in fondo ad essa.")
    st.divider()


def get_user_filename(username):
    return f"chat_history/chat_history_{username}.json"

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

USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    users = load_users()
    if username in users:
        return False  # esiste già
    users[username] = hash_password(password)
    save_users(users)
    return True

def authenticate_user(username, password):
    users = load_users()
    if username in users and users[username] == hash_password(password):
        return True
    return False

