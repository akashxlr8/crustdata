from langgraph.graph import START, StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .state import AgentState, InputState
from datetime import datetime
from langchain_core.messages import RemoveMessage
import json
import streamlit as st
from langchain_core.runnables import RunnableConfig
from loguru import logger
from .configuration import AgentConfiguration, vector_store
from ..common.utils import load_chat_model

from dotenv import load_dotenv
load_dotenv()

logger.add("logs/loguru.log", rotation="100kb")

def entry(state: InputState) -> AgentState:
    """Entry point adds the current message to conversation history"""
    # Get existing messages from input state
    current_messages = list(state.messages)
    
    # Add current user message if not already in history
    if not current_messages or current_messages[-1].content != state.user_message:
        current_messages.append(HumanMessage(content=state.user_message))
    
    # Keep only last 4 messages
    if len(current_messages) > 4:
        current_messages = current_messages[-4:]
        
    logger.info(f"Messages in entry node: {[msg.content for msg in current_messages]}")
    
    return AgentState(
        user_message=state.user_message,
        messages=current_messages
    )

def process_and_generate(state: AgentState, *, config: RunnableConfig) -> AgentState:
    """
    Combined function that handles retrieval, augmentation, and response generation.
    1. Retrieves relevant documents
    2. Augments with context
    3. Generates response using LLM
    """
    # Single log for initial messages
    logger.info(f"Initial messages in process_and_generate: {[msg.content for msg in state.messages]}")
    
    # Retrieval step
    query = str(state.user_message)
    retrieved_docs = vector_store.similarity_search(query)
    
    # Get conversation history
    conversation_history = "\nConversation History:\n"
    for msg in state.messages[-3:]:  # Last 3 messages
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation_history += f"{role}: {msg.content}\n"
    
    # Augmentation step with conversation history
    docs_content = "\n\n".join(
        f"--------Heading: Chunk number {i+1}---------\nContent: {doc.page_content}" 
        for i, doc in enumerate(retrieved_docs)
    )
    
    configuration = AgentConfiguration.from_runnable_config(config)
    system_message = configuration.main_graph_system_prompt.format(
        context=docs_content,
        conversation_history=conversation_history,
        question=state.user_message
    )
    
    # Generation step with conversation history
    messages = [
        SystemMessage(content=system_message),
        *state.messages[-3:]  # Keep last 3 messages for context
    ]
    
    llm = load_chat_model(configuration.query_model)
    response = llm.invoke(messages)
    logger.info(f"Response in process_and_generate: {response}")
    
    # Create AI message
    ai_message = AIMessage(content=str(response.content) if hasattr(response, 'content') else str(response))
    updated_messages = [*state.messages, ai_message]
    
    # Keep only last 4 messages
    if len(updated_messages) > 4:
        updated_messages = updated_messages[-4:]
    
    # Single log for final messages
    logger.info(f"Final messages in process_and_generate: {[msg for msg in updated_messages]}")
    
    return AgentState(
        user_message=state.user_message,
        messages=updated_messages,
        context=state.context,
        answer=str(ai_message.content)
    )

# Graph construction
graph_builder = StateGraph(AgentState)

# Add nodes
graph_builder.add_node("entry", entry)
graph_builder.add_node("process_and_generate", process_and_generate)

# Add edges in sequence
graph_builder.add_edge(START, "entry")
graph_builder.add_edge("entry", "process_and_generate")
graph_builder.add_edge("process_and_generate", END)

# Compile the graph
graph = graph_builder.compile()

def log_query_details(result: dict):
    timestamp = datetime.now().isoformat()
    
    log_entry = {
        "timestamp": timestamp,
        "question": result["question"],
        "answer": result["answer"],
        "context": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in result["context"]
        ]
    }
    
    # Log as JSON for better structure and readability
    logger.info(json.dumps(log_entry, indent=2))

