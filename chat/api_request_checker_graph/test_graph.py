import asyncio
from langchain_core.runnables import RunnableConfig
from chat.api_request_checker_graph.graph import graph
from chat.api_request_checker_graph.state import ApiRequestCheckerState
from langchain_core.messages import HumanMessage, AIMessage
from chat.common.logging_config import get_logger

logger = get_logger("chat.api_request_checker_graph.test_graph")

def test_api_request_checker_graph():
    # Create a sample state
    state = ApiRequestCheckerState(
        message_content="""
        help me with my api request if its correct : curl -X 'GET' \
  'https://reqres.in/api/{resource}?page=1&pe_page=2' \
  -H 'accept: application/json'
        
        """
    )   
    
    # Create a runnable configuration
    config = RunnableConfig()

    try:
        # Invoke the graph
        result =  graph.invoke(state, config=config)
        
        logger.info(f"graph Result: {result}")
        logger.info(f"graph Result: {result[""]}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    test_api_request_checker_graph() 