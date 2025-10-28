import streamlit as st
import time
import mainllm

st.write("Benvenuto nel chatbot di BPM!\nPosso aiutarti nella ricerca di informazioni basandomi sulle RI già sviluppate in precedenza dai tuoi colleghi, facendoti risparmiare molto tempo ;)")

st.caption("Controlla sempre i risultati ottenuti andando a controllare le RI utilizzat per generare la risposta, perché posso sbagliare anche io...")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Di cosa hai bisogno? 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = mainllm.llm_call(prompt)

        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})