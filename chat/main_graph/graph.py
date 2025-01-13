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
from ..common.logging_config import get_logger
from dotenv import load_dotenv
load_dotenv()

logger = get_logger(__name__)
logger.success("Logger initialized in main_graph")

def entry(state: InputState) -> AgentState:
    """Entry point adds the current message to conversation history"""
    # Get existing messages from input state
    logger.info(f"in main_graph entry node called with state: {state}")
    current_messages = list(state.messages)
    
    # Add current user message if not already in history
    if not current_messages or current_messages[-1].content != state.user_message:
        current_messages.append(HumanMessage(content=state.user_message))
    
    logger.info(f"Current messages after adding user message: {[msg.content for msg in current_messages]}")
    # Keep only last 4 messages
    if len(current_messages) > 4:
        current_messages = current_messages[-4:]
    
    logger.info(f"Current messages after trimming: {[msg.content for msg in current_messages]}")
        
    logger.info(f"Returning from entry node with state: {AgentState(user_message=state.user_message, messages=current_messages)}")
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
    logger.debug(f"Initial messages in process_and_generate: {[msg.content for msg in state.messages]}")
    logger.info(f"Processing and generating response for user message: {state.user_message}")
    
    # Retrieval step
    query = str(state.user_message)
    logger.info(f"Querying vector store with query: {query}")
    retrieved_docs = vector_store.similarity_search(query)
    logger.debug(f"Retrieved documents: {[doc.page_content for doc in retrieved_docs]}")
    # Get conversation history
    conversation_history = "\nConversation History:\n"
    for msg in state.messages[-3:]:  # Last 3 messages
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation_history += f"{role}: {msg.content}\n"
    logger.debug(f"Conversation history: {conversation_history}")
    
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
    logger.debug(f"System message in process_and_generate: {system_message}")

    # Generation step with conversation history
    messages = [
        SystemMessage(content=system_message),
        *state.messages[-3:]  # Keep last 3 messages for context
    ]
    
    llm = load_chat_model(configuration.query_model)
    response = llm.invoke(messages)
    logger.debug(f"Response in process_and_generate: {response}")
    logger.info(f"Response receieved from LLM in process_and_generate")
    
    # Create AI message
    ai_message = AIMessage(content=str(response.content) if hasattr(response, 'content') else str(response))
    updated_messages = [*state.messages, ai_message]
    
    # Keep only last 4 messages
    if len(updated_messages) > 4:
        updated_messages = updated_messages[-4:]
    
    # Single log for final messages
    logger.debug(f"Final messages in process_and_generate: {[msg for msg in updated_messages]}")
    
    logger.debug(f"Returning from process_and_generate with state: {AgentState(user_message=state.user_message, messages=updated_messages, context=state.context, answer=str(ai_message.content))}")
    logger.info(f"Returning from process_and_generate successfully")
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

# def log_query_details(result: dict):
#     timestamp = datetime.now().isoformat()
    
#     log_entry = {
#         "timestamp": timestamp,
#         "question": result["question"],
#         "answer": result["answer"],
#         "context": [
#             {
#                 "content": doc.page_content,
#                 "metadata": doc.metadata
#             } for doc in result["context"]
#         ]
#     }
    
#     # Log as JSON for better structure and readability
#     logger.info(json.dumps(log_entry, indent=2))

