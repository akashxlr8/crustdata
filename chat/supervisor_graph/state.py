from pydantic import BaseModel
from typing import List, Optional
from langchain_core.messages import BaseMessage, AnyMessage
from typing_extensions import Annotated
from langgraph.graph import add_messages

class SupervisorState(BaseModel):
    """State for the supervisor graph"""
    messages: Annotated[list[AnyMessage], add_messages]
    current_agent: Optional[str] = None
    final_answer: Optional[str] = None 