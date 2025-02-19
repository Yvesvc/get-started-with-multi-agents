from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from openai_client import get_openai_client
from langgraph.graph import MessagesState


"""
The following example shows you how to create an agent.
An agent is a large language model that has access to tools and can perform actions.

In this example, the tools are defined in a ToolNode, to show you how tools work in LangGraph.
However, LangGraph also provides a ReAct agent out-of-the-box which is a more convenient way to define agents (see example 4_react_agent.py) 

Documentation:
- https://langchain-ai.github.io/langgraph/how-tos/tool-calling/

"""

# First, we define some tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def square(a: int) -> int:
    """Calculates the square of a number."""
    a = int(a)
    return a * a

# The tools are added to a ToolNode
tool_node = ToolNode(tools=[add, multiply, square])

model = get_openai_client()

# The tools are added to the LLM, this lets the LLM know the correct JSON format to call the tools.
llm_with_tools = model.bind_tools([add, multiply, square])

graph_builder = StateGraph(MessagesState)

# We our chatbot node
def chatbot(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# We add the nodes for the chatbot and the tool.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge("tools", END)
graph_builder.add_edge(START, "chatbot")

# Now we add a conditional edge, this allows you to define some logic on how to traverse the graph:
# If the chatbot node indicates that it wants to call a tool, the graph will route to the tools node.
# Else, the graph will route to the END.
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# After the tools node is executed, the graph will route back to the chatbot node.
graph_builder.add_edge("tools", "chatbot")


graph = graph_builder.compile()

user_input = "What is 2 + 2?"
for event in graph.stream({"messages": [("user", user_input)]}, stream_mode="updates"):
    for value in event.values():
        value["messages"][-1].pretty_print()
