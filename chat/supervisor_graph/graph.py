from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

from chat.common.logging_config import get_logger
logger = get_logger("chat.supervisor_graph.graph")

from langchain_core.tools import Tool
from chat.main_graph.graph import graph as rag_graph
from chat.main_graph.state import InputState
#tool for querying a knowledge base
rag_tool = Tool(
    name="knowledge_base_query",
    description="Use this to query a knowledge base of Crustdata API documentation. If the user asks for a specific api request, use this tool to query the api documentation.",
    func=lambda query: rag_graph.invoke(
        InputState(
            user_message=query,
            messages=[]
        )
    )
)

from chat.api_request_checker_graph.graph import graph as api_request_checker_graph
from chat.api_request_checker_graph.state import ApiRequestCheckerState_Input
#tool for verifying api request
api_request_checker_tool = Tool(
    name="api_request_checker",
    description="Use this to verify if the api request is correct. If the user asks to verify a specific api request, use this tool to verify the api request.",
    func=lambda message_content: api_request_checker_graph.invoke(
        ApiRequestCheckerState_Input(
            message_content=message_content
        )
    )
)

tavily_tool = TavilySearchResults(max_results=5)

# This executes code locally, which can be unsafe
repl = PythonREPL()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str

from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import MessagesState, END
from langgraph.types import Command

import operator
class State(TypedDict):
    messages: Annotated[list[dict], operator.add]
    next: str


members = ["knowledge_base_query_agent", "api_request_checker_agent"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal["knowledge_base_query_agent", "api_request_checker_agent", "FINISH"]


llm = ChatOpenAI(model="gpt-4o-mini")


def supervisor_node(state: State) -> Command[Literal["knowledge_base_query_agent", "api_request_checker_agent"]]:
    messages = [
        {"role": "system", "content": system_prompt},
        ] + state["messages"]
    
    logger.info(f"Calling supervisor node with messages: {messages}")
    try:
        response = llm.with_structured_output(Router).invoke(messages)
        logger.info(f"Response from supervisor node: {response}")
        goto = response["next"]
        
        if goto == "FINISH":
            logger.info(f"Reached FINISH state")
            goto = END
        else:
            logger.info(f"Routing to {goto} agent")

        return Command(goto=goto, update={'next': goto})
    except Exception as e:
        logger.error(f"Error in supervisor node: {e}")
        raise 


from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent


knowledge_base_query_agent = create_react_agent(
    llm, tools=[rag_tool], state_modifier="You are a knowledge base query agent, Help users with any queries about Crustdata API documentation."
)

api_request_checker_agent = create_react_agent(
    llm, tools=[api_request_checker_tool], state_modifier="You are a api request checker agent. Check if the api request is correct."
)

def knowledge_base_query_node(state: State) -> Command[Literal["supervisor"]]:
    logger.debug(f" knowledge base query agent recieved state: {state}")
    logger.info(f"In knowledge base query node")
    try:
        result = knowledge_base_query_agent.invoke(state)
        logger.debug(f"knowledge base query agent result: {result}")
        logger.info(f"Knowledge base query result received")
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="knowledge_base_query_agent")
                ]
            },
            goto="supervisor"
        )
    except Exception as e:
        logger.error(f"Error in knowledge base query agent: {e}")
        raise

def api_request_checker_node(state: State) -> Command[Literal["supervisor"]]:
    logger.debug(f"api request checker agent recieved state: {state}")
    logger.info(f"In api request checker node")
    try:
        result = api_request_checker_agent.invoke(state)
        logger.debug(f"api request checker agent result: {result}")
        logger.info(f"Api request checker result received")
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="api_request_checker_agent")
                ]
            },
            goto="supervisor"
        )
    except Exception as e:
        logger.error(f"Error in api request checker agent: {e}")
        raise

# def research_node(state: State) -> Command[Literal["supervisor"]]:
#     result = research_agent.invoke(state)
#     return Command(
#         update={
#             "messages": [
#                 HumanMessage(content=result["messages"][-1].content, name="researcher")
#             ]
#         },
#         goto="supervisor",
#     )


# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
# code_agent = create_react_agent(llm, tools=[python_repl_tool])


# def code_node(state: State) -> Command[Literal["supervisor"]]:
#     result = code_agent.invoke(state)
#     return Command(
#         update={
#             "messages": [
#                 HumanMessage(content=result["messages"][-1].content, name="coder")
#             ]
#         },
#         goto="supervisor",
#     )


builder = StateGraph(State)  # Use State instead of MessagesState

builder.add_node("supervisor", supervisor_node)
# builder.add_node("researcher", research_node)
# builder.add_node("coder", code_node)
builder.add_node("knowledge_base_query_agent", knowledge_base_query_node)
builder.add_node("api_request_checker_agent", api_request_checker_node)

builder.add_edge(START, "supervisor")
builder.add_edge("supervisor", END)

graph = builder.compile()

for s in graph.stream(
    {
        "messages": [
            (
                "user",
                """
               I am trying to use the company screening API for amazon.com but there seems to be some error, can you help me with this api erquest, fix it for me please: curl 'https://api.crustdata.com/screener/identify/' \
    --header 'Accept: application/json, text/plain, */*' \
    --header 'Accept-Language: en-US,en;q=0.9' \
    --header 'Authorization: Token $api_token' \
    --header 'Connection: keep-alive' \
    --header 'Content-Type: application/json' \
    --header 'Origin: https://google.com' \
    --data '{"query_company_website": "amazon.com", "count": 1}'
                """,
            )
        ]
    },
    subgraphs=True,
):
    print(s)
    print("----")
logger.info(f"Graph stream completed")
logger.debug(f"result: {s}")