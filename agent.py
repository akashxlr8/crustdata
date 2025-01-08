from chat.main_graph.graph import graph
from chat.main_graph.state import InputState
from langchain_core.runnables import RunnableConfig
from loguru import logger

def generate_response(message: str) -> str:
    """
    Generate a response using the RAG graph
    """
    # Create input state from user message
    state = InputState(user_message=message)
    
    # Create config
    config = RunnableConfig()
    
    try:
        # Invoke graph and get result
        result = graph.invoke(state, config=config)
        logger.info(f"Graph Result: {result}")
        
        # Extract answer from the result
        if isinstance(result, dict) and "answer" in result:
            logger.info(f"Returning answer: {result['answer']}")
            return result["answer"]
        else:
            logger.error(f"Unexpected result format: {result}")
            return "I apologize, but I encountered an unexpected response format. Please try again."
            
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        return f"I apologize, but I encountered an error while processing your request: {str(e)}" 