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
    # Create input state with user_message
    state = InputState(user_message=message)
    
    try:
        result = graph.invoke(state, config=RunnableConfig())
        logger.info(f"Graph Result: {result}")
        
        # Access the response field from OutputState
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