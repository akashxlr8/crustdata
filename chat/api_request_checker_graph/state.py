from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage, AnyMessage
from typing_extensions import Annotated
from langgraph.graph import add_messages

class ApiRequestCheckerState_Input(BaseModel):
    """State for the api request checker graph"""
    message_content: str 
    

class ApiRequestCheckerState(BaseModel):
    """State for the api request checker graph"""
    message_content: Optional[str] = None
    is_valid_api_request: Optional[bool] = False
    api_validation_result: Optional[str] = None
    api_request: Optional[Dict[str, Any]] = None

class ApiRequestCheckerState_Output(BaseModel):
    """State for the api request checker graph"""
    is_valid_api_request: Optional[bool] = False 
    api_validation_result: Optional[str] = None
    api_request: Optional[Dict[str, Any]] = None
