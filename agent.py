from chat.main_graph.graph import graph
from chat.main_graph.state import InputState, AgentState
from langchain_core.runnables import RunnableConfig
from loguru import logger
from langchain_core.messages import HumanMessage, AIMessage

# Configure logger
logger.add("agent.log", rotation="500 MB", level="INFO")

def generate_response(message: str) -> str:
    """
    Generate a response using the RAG graph
    """
    # Get existing messages from session state if available
    if not hasattr(generate_response, "message_history"):
        generate_response.message_history = []
    
    # Create input state with user_message and history
    state = InputState(
        user_message=message,
        messages=generate_response.message_history
    )
    
    try:
        result = graph.invoke(state, config=RunnableConfig())
        logger.info(f"Graph Result: {result}")
        
        # Update message history
        if isinstance(result, AgentState):
            generate_response.message_history = result.messages
            return result.answer or ""
        elif isinstance(result, dict) and "answer" in result:
            return result["answer"] or ""
        else:
            logger.error(f"Unexpected result format: {type(result)} - {result}")
            return "I apologize, but I encountered an unexpected response format."
            
    except Exception as e:
        logger.exception(f"Error in generate_response: {str(e)}")
        return f"I apologize, but I encountered an error: {str(e)}"