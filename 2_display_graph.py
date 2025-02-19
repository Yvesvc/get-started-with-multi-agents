from langgraph.graph import StateGraph, START, END
from openai_client import get_openai_client
from langgraph.graph import MessagesState

"""
The following example shows how to visualize the graph.

This is particularly useful when you want to make sure that the graph is correctly defined.

Documentation:
https://langchain-ai.github.io/langgraph/how-tos/visualization/#set-up-graph

"""

# Instantiate the OpenAI client
model = get_openai_client()

# We will create the same graph as in the previous example
graph = StateGraph(MessagesState)

def chatbot(state: MessagesState):
    return {"messages": [model.invoke(state["messages"])]}

graph.add_node("chatbot", chatbot)

graph.add_edge(START, "chatbot")

graph.add_edge("chatbot", END)

graph = graph.compile()

try:
    # Draw the graph
    graph_image = graph.get_graph().draw_mermaid_png()
    # Save the graph image
    with open("graph_image.png", "wb") as f:
        f.write(graph_image)
except Exception:
    pass