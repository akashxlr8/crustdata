from pydantic import BaseModel
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, BaseMessage
from typing_extensions import Annotated
from langgraph.graph import add_messages

class InputState(BaseModel):
    """The input state for the main graph."""
    user_message: str

class AgentState(InputState):
    """The state for the main graph."""
    messages: Annotated[list[AnyMessage], add_messages] = []
    context: List[Document] = []
    answer: str = ""
    system_message: Optional[str] = None
    
class OutputState(BaseModel):
    """The output state for the main graph."""
    answer: str 
