from dotenv import load_dotenv

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.tools import tool as Tool
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal
from langgraph.graph import StateGraph, START, END
import json
from datetime import datetime


load_dotenv()

def json_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def llm_call(state: dict, model_with_tools):
    """LLM decides whether to call a tool or not"""

    print("\n--- LLM CALL ---")
    print("LLM calls so far:", state.get("llm_calls", 0))


    print("-----Context provided to LLM:", state["messages"])
    response = model_with_tools.invoke(
        [SystemMessage(content=f"""You are a helpful banking KYC assistant. Chat context:""")]
        + state["messages"]
    )

    print("LLM Response:", response)

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def tool_node(state: dict, tools_by_name: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        if not isinstance(observation, str):
            observation = json.dumps(observation, indent=2, default=json_serializer)
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        print(f"Tool called: {tool.name} with args {tool_call['args']} produced observation: {observation}")
    return {
        "messages": result
    }

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an actioni
    if state["llm_calls"] > 5:
        return END
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

def create_agent(model, tools):
    """Helper function to create the agent"""
    # Augment the LLM with tools
    tools_by_name = {}
    normalized_tools = []

    for tool in tools:
        if isinstance(tool, Tool):
            normalized_tool = tool
        elif callable(tool):
            tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
            if tool_name is None:
                raise ValueError(f"Cannot determine tool name for {tool!r}")
            normalized_tool = Tool.from_function(func=tool, name=tool_name, description=(tool.__doc__ or ""))
        else:
            raise TypeError(f"Tool must be a langchain Tool or callable, got {type(tool)}")

        normalized_tools.append(normalized_tool)
        tools_by_name[normalized_tool.name] = normalized_tool

    model_with_tools = model.bind_tools(normalized_tools)

    # Build workflow
    agent_builder = StateGraph(MessagesState)

    # Add nodes
    agent_builder.add_node("llm_call", lambda state: llm_call(state, model_with_tools))
    agent_builder.add_node("tool_node", lambda state: tool_node(state, tools_by_name))

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")

    # Compile the agent
    print("Compiling agent...")
    agent = agent_builder.compile()
    return agent