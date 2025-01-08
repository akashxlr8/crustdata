from pydantic import BaseModel
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AnyMessage
from typing_extensions import Annotated
from langgraph.graph import add_messages

class InputState(BaseModel):
    """The input state contains the current user message and optional message history"""
    user_message: str
    messages: Annotated[list[AnyMessage], add_messages]  # Conversation history

    def model_post_init(self, *args, **kwargs):
        """Post initialization hook to trim messages"""
        super().model_post_init(*args, **kwargs)
        # Keep only last 10 messages
        if len(self.messages) > 10:
            self.messages = self.messages[-10:]
            
class AgentState(BaseModel):
    """The state for the main graph."""
    user_message: str
    messages: Annotated[list[AnyMessage], add_messages]  # Conversation history
    context: List[Document] = []
    system_message: Optional[str] = None
    answer: Optional[str] = None

    # def add_message(self, message: BaseMessage) -> None:
    #     """Add a message to the conversation history"""
    #     self.messages.append(message)
    #     # Keep only last 4 messages
    #     if len(self.messages) > 4:
    #         self.messages = self.messages[-4:]
    
    def model_post_init(self, *args, **kwargs):
        """Post initialization hook to trim messages"""
        super().model_post_init(*args, **kwargs)
        # Keep only last 10 messages
        if len(self.messages) > 10:
            self.messages = self.messages[-10:]
