import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing_extensions import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, max_retries=2)

class AgentState(TypedDict):
  messages : List[HumanMessage]
  
def answer_node(state : AgentState) -> AgentState:
  """Node to generate AI response"""
  response = llm.invoke(state['messages'])
  print(f"\n AI response : {response.content}")
  return state

graph_builder = StateGraph(AgentState)
graph_builder.add_node("generate_answer" , answer_node)
graph_builder.add_edge(START , "generate_answer")
graph_builder.add_edge("generate_answer" , END)

graph = graph_builder.compile()

user_input = input("Enter : ")
while user_input != "exit":
  graph.invoke({"messages" : [HumanMessage(content=user_input)]})
  user_input = input("Enter : ")

print("goodbye!")