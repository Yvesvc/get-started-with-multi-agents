from langgraph.graph import StateGraph, START, END
from openai_client import get_openai_client
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
import uuid

"""
So far we have seen how to create a chatbot that can interact with the user and use tools.
However, the state of the conversation is not saved between invocations.

The following example shows how to save the state (chat history + steps + tool calls + ...) in memory.

This is particularly useful in situations where you want to
- save the state of the conversation (eg for later analysis).
- resume the conversation at a later point. 

If you want more persistent storage, you should use a database like Postgres.

Documentation:
- https://langchain-ai.github.io/langgraph/concepts/persistence/
- https://langchain-ai.github.io/langgraph/how-tos/persistence/
- https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres/

"""

# As usual, we define our graph

model = get_openai_client()

def chatbot(state: MessagesState):
    return {"messages": [model.invoke(state["messages"])]}

graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# We compile the graph, but this time we add a MemorySaver checkpointer.
# This object will save the state of the conversation in memory.
graph = graph.compile(checkpointer=MemorySaver())

# We can now invoke the graph, but we need a way to identify the conversation (if we want to resume it later).
# This is done by setting a thread_id in the config.
my_unique_conversation_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": my_unique_conversation_id}}

response = graph.invoke({"messages": [("human", "My name is Bob")]}, config)
for message in response["messages"]:
    message.pretty_print()

# you can also get the state of the conversation at any point using the get_state method
snapshot = graph.get_state(config)

print("\n")
print("RESUMING THE CONVERSATION (STATE IS RESTORED)")
print("\n")

# Because the state is saved in memory and we are using the same thread id, the LLM remembers your name.
response = graph.invoke({"messages": [("human", "What is my name?")]}, config)
for message in response["messages"]:
    message.pretty_print()