""" This is the graph for the api request checker. It is used to verify if the api request provided to the user is correct. 
It is used in the supervisor graph to verify if the api request provided to the user is correct.
if the LLM response involves any api request, it is passed to this graph to verify if the api request is correct by calling the api endpoint,
checking if the api request is valid and if the api request is successful.
If not successful, We fix the api request and pass it to the supervisor graph to be used in the next iteration.
"""

from langgraph.graph import START, StateGraph, END
from requests.exceptions import RequestException
from .state import ApiRequestCheckerState, ApiRequestCheckerState_Input, ApiRequestCheckerState_Output
import re
import requests
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from chat.common.logging_config import get_logger

logger = get_logger("chat.api_request_checker_graph.graph")

def entry(state: ApiRequestCheckerState_Input) -> ApiRequestCheckerState:
    """Entry point for the api request checker graph"""
    logger.info(f"Entered into the api request checker graph")
    #get the message content from the state
    message_content = state.message_content
    logger.debug(f"Message content received in the entry node: {message_content}")
 
    
    return ApiRequestCheckerState(
        message_content=message_content,
        is_valid_api_request=None,
        api_validation_result=None,
        api_request=None
    )

def process_message(state: ApiRequestCheckerState) -> ApiRequestCheckerState:
    """
    Process the message and return the state
    """
    logger.info(f"Entered into the process message node")
    logger.debug(f"State received in the process message node: {state}")
    #get the message content from the state
    message_content = state.message_content
    
    # Ensure message_content is a string
    if message_content is None:
        message_content = ""
    
    logger.debug(f"Message content received in the process message node: {message_content}")
    # Extract the api request from the user message
    api_request = extract_api_request(message_content) or ""
    logger.debug(f"Api request received in the process message node after extraction: {api_request}")
    
    # Parse the api request
    api_request = parse_curl_to_request(api_request)
    logger.debug(f"Api request received in the process message node after parsing: {api_request}")
    
    #check if the user message is a valid api request
    if validate_api_request(api_request):
        logger.debug(f"Api request is valid")
        logger.info(f"Exiting the process message node")
        logger.debug(f"returning the state from the process message node: {state}")
        return state.model_copy(update={"is_valid_api_request": True, "api_request": api_request})
    else:
        logger.debug(f"Api request is not valid")
        
        # If API request failed, use LLM to fix it
        llm = ChatOpenAI(model="gpt-4o-mini")
        fix_prompt = SystemMessage(content="""
        You are an API request fixer. Given a failed API request and error message,
        analyze the issue and provide a corrected version of the request.
        Only output the corrected curl command, nothing else.
        """)
        
        error_context = HumanMessage(content=f"""
        Failed API request: {api_request}
        Error: {state.api_validation_result}
        Please fix this request.
        """)
    
        fixed_request = llm.invoke([fix_prompt, error_context])
        logger.debug(f"Fixed request received in the process message node: {fixed_request}")
        logger.info(f"Exiting the process message node")
        logger.debug(f"returning the state from the process message node after fixing the api request: {state}")
        return state.copy(update={
            "is_valid_api_request": True,
            "api_validation_result": "API request fixed",
            "api_request": fixed_request.content
            })

def extract_api_request(text: str) -> Optional[str]:
    """
    Extract the api request from the text
    """
    curl_pattern = r'curl[^`]*?(?=\n|$)'
    matches = re.findall(curl_pattern, text, re.DOTALL)
    logger.debug(f"Matches received in the extract api request node: {matches}")
    if matches:
        return matches[0].strip()
    logger.info(f"No matches received in the extract api request node")
    return None

def parse_curl_to_request(curl_request: str) -> dict:
    """
    Parse the curl request to a dictionary
    """
    logger.info(f"Entered into the parse curl to request node")
    logger.debug(f"Curl request received in the parse curl to request node: {curl_request}")
    
    headers = {}
    data = None
    
    url_match = re.search(r"'(https?://[^']+)'", curl_request)
    url = url_match.group(1) if url_match else ""
    
    logger.debug(f"Url received in the parse curl to request node: {url}")
    
    #extract headers
    header_matches = re.finditer(r"-H '([^:]+): ([^']+)'", curl_request)
    for match in header_matches:
        headers[match.group(1).strip()] = match.group(2).strip()
    logger.debug(f"Headers received in the parse curl to request node: {headers}")
    
    
    #extract data
    data_matches = re.finditer(r"--data '([^']+)'", curl_request)
    for match in data_matches:
        data = match.group(1).strip()
        
    logger.debug(f"Data received in the parse curl to request node: {data}")
    logger.info(f"Exiting the parse curl to request node")
    logger.debug(f"returning the request from the parse curl to request node: {{'url': {url}, 'headers': {headers}, 'data': {data}}}")

    return {"url": url, "headers": headers, "data": data}

def validate_api_request(request: dict) -> dict:
    """
    Validate the api request by calling the api endpoint and checking if the api request is successful
    """
    logger.info(f"Entered into the validate api request node")
    logger.debug(f"Request received in the validate api request node: {request}")
    #call the api endpoint
    try:
        logger.debug(f"Calling the api endpoint")
        response = requests.post(request["url"], headers=request["headers"], data=request["data"])
        logger.debug(f"Response received in the validate api request node: {response}")
        if response.status_code == 200:
            logger.info(f"API request successful")
            logger.info(f"Exiting the validate api request node")
            logger.debug(f"returning the response from the validate api request node: {response}")
            return {"status": "success", "message": "API request successful"}
        else:
            logger.info(f"API request failed")
            logger.info(f"Exiting the validate api request node")
            logger.debug(f"returning the response from the validate api request node: {response}")
            return {"status": "failure", "message": "API request failed"}
    except Exception as e:
        return {"status": "failure", "message": f"API request failed: {e}"}


def output(state: ApiRequestCheckerState) -> ApiRequestCheckerState_Output:
    logger.info(f"Entered into the output node")
    logger.debug(f"State received in the output node: {state}")
    return ApiRequestCheckerState_Output(
        is_valid_api_request=state.is_valid_api_request,
        api_validation_result=state.api_validation_result,
        api_request=state.api_request
    )

graph = StateGraph(ApiRequestCheckerState)

graph.add_node("entry", entry)
graph.add_edge(START, "entry")

graph.add_node("process_message", process_message)
graph.add_edge("entry", "process_message")

graph.add_node("output", output)
graph.add_edge("process_message", "output")
graph.add_edge("output", END)

graph = graph.compile()