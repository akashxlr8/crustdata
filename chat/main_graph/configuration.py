from dataclasses import dataclass, field
from ..common import prompts
from chat.common.config import BaseConfiguration
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import JinaEmbeddings
from pydantic import SecretStr
import os
from typing import Any

@dataclass(kw_only=True)
class AgentConfiguration(BaseConfiguration):
    """The configuration for the supervisor agent."""
    
    #models
    
    #prompts
    main_graph_system_prompt: str = field(
    default=prompts.Main_graph_system_prompt,
        metadata={
            "description": "The system prompt used for classifying user questions to route them to the correct node."
        },
    )
    
    # general_system_prompt: str = field(
    #     default=prompts.GENERAL_SYSTEM_PROMPT,
    #     metadata={
    #         "description": "The system prompt used for responding to general questions."
    #     },
    # )
    
    # more_info_system_prompt: str = field(
    #     default=prompts.MORE_INFO_SYSTEM_PROMPT,
    #     metadata={
    #         "description": "The system prompt used for asking for more information from the user."
    #     },
    # )

    
    # response_system_prompt: str = field(
    #     default=prompts.RESPONSE_SYSTEM_PROMPT,
    #     metadata={"description": "The system prompt used for generating responses."},
    # )
    
embeddings = JinaEmbeddings(
        jina_api_key=SecretStr(os.getenv("JIN_API_KEY") or "jina_808c30ece3374808af85b4884df6f5eb4kd4UOSI6brj7ZvTE3qB2QfMuHba"), model_name="jina-embeddings-v3", session=Any
    )
vector_store = PineconeVectorStore(index_name="crustdata-docs2", embedding=embeddings)
