import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing_extensions import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, max_retries=2)

class AgentState(TypedDict):
  messages : List[Union[HumanMessage, AIMessage]]

def generate_response(state : AgentState) -> AgentState:
  """Node to generate the response to human message"""
  response = llm.invoke(state["messages"])
  print(response.content)
  state['messages'].append(AIMessage(content=response.content))
  return state

graph_builder = StateGraph(AgentState)
graph_builder.add_node("generate_response", generate_response)
graph_builder.add_edge(START, "generate_response")
graph_builder.add_edge("generate_response", END)
graph = graph_builder.compile()

conversation_history = []

user_input = input("Enter : ")
while user_input != "exit" : 
  conversation_history.append(HumanMessage(content=user_input))
  result = graph.invoke({"messages" : conversation_history})
  print(result['messages'])
  conversation_history = result['messages']
  user_input = input("Enter : ")
  
  
with open("logging.txt" , "w") as file:
  file.write("Your conversation log: \n")
  
  for message in conversation_history:
    if isinstance(message , HumanMessage):
      file.write(f"Human : {message.content}\n")
    elif isinstance(message , AIMessage):
      file.write(f"AI : {message.content}\n")
  file.write("End of conversation")
  
print("Conversation saved to logging.txt file")
print("Goodbye!")
