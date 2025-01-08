from langgraph.graph import START, StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from .state import AgentState, InputState, OutputState
from datetime import datetime
import json

from langchain_core.runnables import RunnableConfig
from loguru import logger
from .configuration import AgentConfiguration, vector_store
from ..common.utils import load_chat_model

from dotenv import load_dotenv
load_dotenv()

logger.add("loguru.log")

def entry(state: InputState):
    """
    This function is the entry point for the graph. It takes the input state and returns the initial state for the graph.
    """
    return {"user_message": state.user_message}

def retrieve(state: AgentState):
    """
    Retrieves the context documents from the vector store based on the user's `question`.
    """
    last_user_message = state.user_message
    retrieved_docs = vector_store.similarity_search(last_user_message)
    return {"context": retrieved_docs}

def augment(state: AgentState, *, config: RunnableConfig):
    """
    Augments the context with the retrieved documents from  the retrieve node.
    And makes the system message with the context and the user's question.
    """
  
    retrieved_docs = state.context
    docs_content = "\n\n".join(
        f"--------Heading: Chunk number {i+1}---------\nContent: {doc.page_content}" 
        for i, doc in enumerate(retrieved_docs)
    )
    configuration = AgentConfiguration.from_runnable_config(config)

    system_message = configuration.main_graph_system_prompt.format(
        context=docs_content,
        question=state.user_message
    )
    return {"system_message": system_message}

def generate(state: AgentState, *, config: RunnableConfig):
    
    
    # Create messages list with system message and the last user message
    
                      
    messages = [
        SystemMessage(content=state.system_message or ""),
        HumanMessage(content=state.user_message)
    ]
    # Log the messages being sent
    logger.info("Messages being sent to LLM:")
    logger.info(messages)
    configuration = AgentConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.response_model)
    # Invoke the LLM with the messages
    response = llm.invoke(messages)
    
    # Log the response
    logger.info(f"LLM Response: {response.content}")
    
    # Return both answer and user_message to maintain state
    return {"answer": response.content, "user_message": state.user_message}

def exit(state: AgentState):
    """This is the exit node for the graph. It takes the AgentState and returns the final state for the graph.
    
    Args:
        state: The AgentState for the graph.
        
    Returns:
        OutputState: The final answer in string format for the graph.
    """
    return OutputState(answer=state.answer)

# Graph construction
graph_builder = StateGraph(AgentState)

# Add nodes
graph_builder.add_node("entry", entry)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("augment", augment)
graph_builder.add_node("generate", generate)
graph_builder.add_node("exit", exit)
# Add edges in sequence
graph_builder.add_edge(START, "entry")
graph_builder.add_edge("entry", "retrieve")
graph_builder.add_edge("retrieve", "augment")
graph_builder.add_edge("augment", "generate")
graph_builder.add_edge("generate", "exit")
graph_builder.add_edge("exit", END)

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

