import streamlit as st
from streamlit.web.server.websocket_headers import _get_websocket_headers

# tag::write_message[]
def write_message(role, content, save = True):
    """
    This is a helper function that saves a message to the
     session state and then writes a message to the UI
    """
    # Append to session state
    if save:
        st.session_state.messages.append({"role": role, "content": content})

    # Write to UI
    with st.chat_message(role):
        st.markdown(content)
# end::write_message[]

# tag::get_session_id[]
def get_session_id():
    headers = _get_websocket_headers()
    return headers.get("Sec-WebSocket-Key") if headers else None
# end::get_session_id[]