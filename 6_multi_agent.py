import random
from langgraph.graph import StateGraph, START, MessagesState
from langchain_core.tools import tool
import random
from typing_extensions import Literal
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from openai_client import get_openai_client
from langchain_core.messages import ToolMessage
from typing import Annotated
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState

"""
So far we have defined rather simple graphs in which one agent interacts with the user.
For more complex tasks, you want to have multiple agents interacting with each other.

The tricky part in multi-agent systems is how you hand off the conversation from one agent to another.
In my experience, the best way to do this is to leverage tool calling.
Why? Because tools are a way to enforce structured output, which makes it easier to do a handover.
To do this, you need to create a special type of tool that will do the handoff.
The agent can then decide to call this tool to hand off the conversation.
The logic of the handoff is defined in the tool.

Documentation
- https://langchain-ai.github.io/langgraph/concepts/multi_agent/
- https://langchain-ai.github.io/langgraph/how-tos/agent-handoffs/

"""



model = get_openai_client()

@tool
def get_travel_recommendations():
    """Get recommendation for travel destinations"""
    return random.choice(["aruba", "turks and caicos"])

@tool
def get_hotel_recommendations(location: Literal["aruba", "turks and caicos"]):
    """Get hotel recommendations for a given destination."""
    return {
        "aruba": [
            "The Ritz-Carlton, Aruba (Palm Beach)"
            "Bucuti & Tara Beach Resort (Eagle Beach)"
        ],
        "turks and caicos": ["Grace Bay Club", "COMO Parrot Cay"],
    }[location]

# this is a special type of tool that will do the handoff to another agent
def make_handoff_tool(*, agent_name: str):
    """Create a tool that can return handoff via a Command"""

    # This tool will be called by an agent to hand off the conversation to another agent
    # the name of the other agent is passed as an argument

    
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def handoff_to_agent(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        
        # A tool message is inserted into the chat history to indicate the handoff
        """Ask another agent for help."""
        tool_message = ToolMessage(
            content = f"Successfully transferred to {agent_name}",
            tool_name = tool_name,
            tool_call_id = tool_call_id,
        )
        # the Command object will do the handoff
        # the goto argument specifies the agent to hand off to
        # because current tool is part of the subgraph of the agent it belongs to, we need to traverse back to the main graph
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={"messages": state["messages"] + [tool_message]},
        )

    return handoff_to_agent

travel_advisor = create_react_agent(
    model,
    # the travel advisor can recommend travel destinations or hand off to the hotel advisor
    [get_travel_recommendations, make_handoff_tool(agent_name="hotel_advisor")],
    prompt=(
        "You are a general travel expert that can recommend travel destinations (e.g. countries, cities, etc). "
        "If you need hotel recommendations, ask 'hotel_advisor' for help. "
        "You MUST include human-readable response before transferring to another agent."
    ),
)

def call_travel_advisor(
    state: MessagesState,
) -> Command[Literal["hotel_advisor", "__end__"]]:
    return travel_advisor.invoke(state)


hotel_advisor = create_react_agent(
    model,
    # the hotel advisor can recommend hotels or hand off to the travel advisor
    [get_travel_recommendations, make_handoff_tool(agent_name="travel_advisor")],
    prompt=(
        "You are a hotel expert that can provide hotel recommendations for a given destination. "
        "If you need help picking travel destinations, ask 'travel_advisor' for help."
        "You MUST include human-readable response before transferring to another agent."
    ),
)

def call_hotel_advisor(
    state: MessagesState,
) -> Command[Literal["travel_advisor", "__end__"]]:
    return hotel_advisor.invoke(state)

graph = StateGraph(MessagesState)
graph.add_node("travel_advisor", call_travel_advisor)
graph.add_node("hotel_advisor", call_hotel_advisor)
graph.add_edge(START, "travel_advisor")
graph = graph.compile()

user_input = "i wanna go somewhere warm in the caribbean. pick one destination and give me hotel recommendations"

event = graph.invoke({"messages": [("user", user_input)]})
for message in event['messages']:
    message.pretty_print()