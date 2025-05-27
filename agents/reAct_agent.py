import os 
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

class AgentState(TypedDict):
  messages : Annotated[Sequence[BaseMessage] , add_messages]
  
@tool
def add(a: int, b: int) -> int:
  """This is an addition function to add two numbers together"""
  return a + b

@tool
def subtract(a: int, b: int) -> int:
  """This is an subtraction function to add two numbers together"""
  return a - b

@tool
def multiply(a: int, b: int) -> int:
  """This is an multiplication function to add two numbers together"""
  return a * b

tools = [add, subtract, multiply]

model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, max_retries=2).bind_tools(tools)

def model_call(state : AgentState) -> AgentState:
  """Node to call the model"""
  system_prompt = SystemMessage(content="Your are my AI assistant. Please answer my query to the best od your ability")
  response = model.invoke([system_prompt] + state["messages"])
  return {"messages" : [response]}

def should_continue(state : AgentState) -> str :
  """This is the node that decides whether the execution should continue or end"""
  messages = state["messages"]
  last_message = messages[-1]
  
  if not last_message.tool_calls:
    return "END"
  else :
    return "continue"
  
graph_builder = StateGraph(AgentState)
graph_builder.add_node("model_call" , model_call)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools" , tool_node)

graph_builder.add_edge(START, "model_call")
graph_builder.add_conditional_edges(
  "model_call",
  should_continue,
  {
    "continue" : "tools",
    "end" : END
  }
)

graph_builder.add_edge("model_call" , "tools")
graph = graph_builder.compile()

def print_stream(stream):
  for s in stream:
    message = s["messages"][-1]
    if isinstance(message , tuple):
      print(message)
    else:
      message.pretty_print()
      
inputs = {"messages": [("user", "Add 40 + 12 and what is 6 plus 7, multiply 3 to the result of addition of 6 and 7 and subtract by 9")]}
print_stream(graph.stream(inputs, stream_mode="values"))