from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
prompt = ChatPromptTemplate.from_messages([
    ("system","You are a AI enginner in Google.answer me in Korean no matter what."),
     ("user","{input}")
])
llm = ChatOllama(model="gemma:latest")

# chain = prompt | llm
# response = chain.invoke({"input":"What is diffusion?"})
# print(response)

chain = prompt | llm | StrOutputParser()
response = chain.invoke({"input": "Why is the rag process for LLM not to work?"})
print(response)