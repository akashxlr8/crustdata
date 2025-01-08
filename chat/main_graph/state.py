from pydantic import BaseModel
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing_extensions import Annotated
from langgraph.graph import add_messages

class InputState(BaseModel):
    """The input state contains only the current user message"""
    user_message: str


class AgentState(InputState):
    """The state for the main graph."""
    messages: Annotated[List[BaseMessage], add_messages] = []  # Conversation history
    context: List[Document] = []
    system_message: Optional[str] = None
    answer: Optional[str] = None
    

