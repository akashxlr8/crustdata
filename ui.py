import streamlit as st
from utils import write_message
from agent import generate_response

# Page Config
st.set_page_config("CrustData Assistant", page_icon="🤖")

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm the CrustData Assistant. How can I help you with the API?"},
    ]

# Submit handler
def handle_submit(message):
    """Submit handler that uses our RAG agent"""
    with st.spinner('Thinking...'):
        # Call the agent
        response = generate_response(message)
        write_message('assistant', response)

# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if prompt := st.chat_input("Ask me about the CrustData API..."):
    # Display user message
    write_message('user', prompt)
    # Generate response
    handle_submit(prompt)