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

    /* Messaggi utente */
    [data-testid="stChatMessage"][data-testid="user"] {
        background-color: #e9ecef;
        float: right;
    }

    /* Messaggi assistant */
    [data-testid="stChatMessage"][data-testid="assistant"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        float: left;
    }
                
    /* Campo di input chat */
    .stChatInputContainer {
        background-color: white !important;
        border-radius: 0.75rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dee2e6;
    }

    /* Hover su input base */  
    div[data-baseweb="input"]:hover > div {
        background-color: #f8f9fa !important;
        border-radius: 0.75rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }

    div[data-baseweb="input"] > div {
        background-color: #f8f9fa !important;
        border: 1.5px solid #dee2e6 !important;
        box-shadow: 0 0 6px rgba(0,123,255,0.25);
        border-radius: 0.5rem !important;
        padding: 0.4rem 0.8rem !important;
        transition: all 0.2s ease-in-out;
    }

    /* Clear float dopo la chat */
    [data-testid="stVerticalBlock"] {
        overflow: auto;
    }
    </style>
    """, unsafe_allow_html=True)

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

