from langgraph.graph import START, StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .state import AgentState, InputState
from datetime import datetime
from langchain_core.messages import RemoveMessage


from langchain_core.runnables import RunnableConfig
from loguru import logger
from .configuration import AgentConfiguration, vector_store
from ..common.utils import load_chat_model

from dotenv import load_dotenv
load_dotenv()

logger.add("loguru.log")

def entry(state: InputState):
    """Entry point adds the current message to conversation history"""
    # Create new state with current message added to history
    return AgentState(
        user_message=state.user_message,
        messages=[*state.messages[-3:], HumanMessage(content=state.user_message)]
    )

def process_and_generate(state: AgentState, *, config: RunnableConfig):
    """
    Combined function that handles retrieval, augmentation, and response generation.
    1. Retrieves relevant documents
    2. Augments with context
    3. Generates response using LLM
    """
    # Retrieval step
    query = str(state.user_message)
    retrieved_docs = vector_store.similarity_search(query)
    
    # Augmentation step
    docs_content = "\n\n".join(
        f"--------Heading: Chunk number {i+1}---------\nContent: {doc.page_content}" 
        for i, doc in enumerate(retrieved_docs)
    )
    
    configuration = AgentConfiguration.from_runnable_config(config)
    system_message = configuration.main_graph_system_prompt.format(
        context=docs_content,
        question=state.user_message
    )
    
    # Generation step with conversation history
    messages = [
        SystemMessage(content=system_message),
        *state.messages[-3:]  # Keep last 3 messages for context
    ]
    
    llm = load_chat_model(configuration.response_model)
    response = llm.invoke(messages)
    logger.info(f"Response in process_and_generate: {response}")
    
    # Create AI message
    ai_message = AIMessage(content=str(response.content) if hasattr(response, 'content') else str(response))
    
    # Return updated state with new message and trimmed history
    return AgentState(
        user_message=state.user_message,
        messages=[*state.messages[-3:], ai_message],  # Keep last 3 messages plus new response
        context=state.context,
        answer=str(ai_message.content)  # Explicitly convert to string
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

