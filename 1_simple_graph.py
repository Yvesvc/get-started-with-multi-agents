from langgraph.graph import StateGraph, START, END
from openai_client import get_openai_client
from langgraph.graph import MessagesState

"""
The following example shows how to create your first graph using LangGraph,
teaching you the building blocks of the framework.

In LangGraph, you define your workflows and agentic systems using a graph.
A graph is a very simple structure consisting of nodes and edges. 
Because the building blocks are so simple, you can create easy but also very advanced systems by combining them in different ways.

Documentation:
- https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-1-build-a-basic-chatbot

"""

# Instantiate the OpenAI client (you can use any other LLM).
model = get_openai_client()

# Start by creating a StateGraph. A StateGraph object defines the structure of our graph.
# The state is stored in the MessagesState object, this is a dictionary containing the current conversation messages
graph = StateGraph(MessagesState)

# Create a node, this is a Python function that invokes an LLM and updates the state.
# It doesn't alwats have to invoke an LLM, it can also call an external API or do some other processing.
# The state is passed as an argument to the node.
def chatbot(state: MessagesState):
    # The LLM is invoked with the current messages in the state.
    response = model.invoke(state["messages"])
    # And the state is updated with the output of the LLM. 
    return {"messages": [response]} 

# Add the node to the graph.
graph.add_node("chatbot", chatbot)

# A graph should always have an entry point, in this case we set the chatbot node as the start node.
graph.add_edge(START, "chatbot")

# A graph should always have an exit point
# In this case we set the chatbot node as the end node. The graph will stop executing when it reaches this node.
graph.add_edge("chatbot", END)

# Finally, compile the graph so that it can be executed.
graph = graph.compile()

user_input = "Hello, how are you?"

# Invoke the graph with the user input.
# The stream method () will return the current state every time it gets updated by a node.
# In this case only one node will be called (chatbot node) so stream() will only return the state once.
for event in graph.stream({"messages": [("user", user_input)]}):
    # We return the last message from the chatbot
    for value in event.values():
        print("Assistant:", value["messages"][-1].content)