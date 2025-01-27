from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph, add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


import os
from typing import Sequence
from typing_extensions import Annotated, TypedDict



class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # language: str
    context: str


# Initialize the language model
llm = ChatAnthropicVertex(
                model_name="claude-3-5-sonnet-v2@20241022",
                project=os.getenv('GOOGLE_CLOUD_PROJECT'),
                location='us-east5',
                temperature=0.1
            )

workflow = StateGraph(state_schema=State)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Use the context provided. {context}."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
runnable = prompt | llm

# Define the function that calls the model
def call_model(state: State):
    response = runnable.invoke(state)
    # Update message history with response:
    return {"messages": [response]}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}
query = "Hi! I'm Bob."


input_messages = {"messages": [HumanMessage(query)], 
                #   "language": "Spanish"
                  "context":"My full name is Bob Smith. use full name when addressing me"
                  }
output = app.invoke(input_messages, config)
output["messages"][-1].pretty_print()  # output contains all messages in state

# Print history before processing new question
print("\n=== Current Conversation History ===")
try:
    # Using app.get_state similar to docs
    current_state = app.get_state(config).values
    print("current_state:", current_state)
    if current_state and "messages" in current_state:
        print("Previous messages:")
        for message in current_state["messages"]:
            message.pretty_print()
        print("---")
    else:
        print("No previous history")
except Exception as e:
    print(f"Error loading history: {e}")
    print("No previous history")

