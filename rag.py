from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
embedding_model = HuggingFaceEmbeddings(model_name = 'jhgan/ko-sbert-nli',
                                        model_kwargs={'device':'cuda'},
                                        encode_kwargs={'normalize_embeddings':True})
query = 'What is specific feature about monkey?'
vectorstore = FAISS.load_local('./db',embedding_model,allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type='mmr',
                                     earch_kwargs={'k': 20,'lambda_mult':0.7})
llm = ChatOllama(model="gemma:latest")
docs = retriever.get_relevant_documents(query)
prompt = ChatPromptTemplate.from_messages([
    ("system","You are a AI enginner in Google.answer me in Korean no matter what."),
     ("user","{input}")
])

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])
   
chain = prompt | llm | StrOutputParser()
response = chain.invoke({'input':(format_docs(docs))})    
print(response)