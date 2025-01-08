from chat.main_graph.graph import graph
from chat.main_graph.state import InputState, AgentState
from langchain_core.runnables import RunnableConfig
from loguru import logger
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st

# Configure logger
logger.add("agent.log", rotation="500 MB", level="INFO")
def generate_response(message: str) -> str:
    """Generate a response using the RAG graph"""
    # Convert Streamlit session state messages to LangChain messages
    current_messages = []
    
    # Log Streamlit state before conversion
    logger.info(f"Streamlit session state messages: {st.session_state.messages}")
    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            current_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            current_messages.append(AIMessage(content=msg["content"]))
    
    # Log converted messages
    logger.info(f"Converted LangChain messages: {[f'{msg.type}: {msg.content}' for msg in current_messages]}")    
    # Create input state with user_message and converted messages
    state = InputState(
        user_message=message,
        messages=current_messages
    )
    
    try:
        result = graph.invoke(state, config=RunnableConfig())
        logger.info(f"Graph Result: {result}")
        
        if isinstance(result, AgentState):
            return result.answer or ""
        elif isinstance(result, dict) and "answer" in result:
            return result["answer"] or ""
        else:
            logger.error(f"Unexpected result format: {type(result)} - {result}")
            return "I apologize, but I encountered an unexpected response format."
            
    except Exception as e:
        logger.exception(f"Error in generate_response: {str(e)}")
        return f"I apologize, but I encountered an error: {str(e)}"