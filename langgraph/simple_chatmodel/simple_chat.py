from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import START,END,StateGraph
from typing import Annotated
from pydantic import BaseModel
from fastapi import FastAPI



app = FastAPI('')

@app.post('/')
async def llama(question : str):
    
    llm = ChatOllama(model='llama3.1:latest',temperature=0)
    response = llm.invoke(question)
    


