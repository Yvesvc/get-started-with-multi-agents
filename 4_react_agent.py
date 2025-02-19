import random
from typing_extensions import Literal
from langgraph.graph import StateGraph, START
from typing_extensions import Literal
from openai_client import get_openai_client
from langchain_core.tools import tool
import random
from typing_extensions import Literal
from langgraph.graph import StateGraph, START
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent

"""
As shown in the previous example, you can create an agent yourself, but LangGraph also provides a pre-built agent called ReAct.
The React agent is a more convenient way to define agents.

Documentation:
- https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/

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


# Defining a ReAct agent is as simple as calling create_react_agent with the model, tools, and prompt.
travel_advisor = create_react_agent(
    model,
    [get_travel_recommendations, get_hotel_recommendations],
    prompt=(
        "You are a general travel expert that can recommend travel destinations (e.g. countries, cities, etc). "
    ),
)

# Define the call_travel_advisor node that will call the travel_advisor agent
# The ReAct agent will also automatically update the state
def call_travel_advisor(state: MessagesState):
    return travel_advisor.invoke(state)

graph = StateGraph(MessagesState)
graph.add_node("travel_advisor", call_travel_advisor)
graph.add_edge(START, "travel_advisor")
graph = graph.compile()

user_input = "What hotel do you recommend in Aruba?"

for event in graph.stream({"messages": [("user", user_input)]}):
    for value in event.values():
        for message in value["messages"]:
            message.pretty_print()
