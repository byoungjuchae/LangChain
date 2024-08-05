from langgraph.graph import StateGraph, add_messages
from langchain_ollama import ChatOllama
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.tools.tavily_search import TavilySearchResults
import os


os.environ['TAVILIY_SEARCH_API'] = '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'
tool = TavilySearchResults(max_results=2)

memory = SqliteSaver.from_conn_string(":memory:")
class State(TypedDict):
    
    messages : Annotated[ist, add_messages]
    

llm = ChatOllama(model='llama3.1:latest',temperature=0)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    
    return {'messages':[llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot",chatbot)

tools = ToolNode(tool)
graph_builder.add_node("tools",tools)
graph_builder.add_conditional_edges("chatbot",tools_condition)
graph_builder.add_edge("tools","chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpoint=memory)