from langchain_community.document_loaders import WebBaseLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



loader = WebBaseLoader('https://python.langchain.com/v0.2/docs/introduction/')

docs = loader.load()

text_split = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=10)

docs = text_split.split_documents(docs)
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-l6-v2',
                                        model_kwargs={'device':'cuda'})

vectordb = FAISS.from_documents(docs,
                                embedding_model
                                )
retriever = vectordb.as_retriever(search_type="similarity_score_threshold",
                                 search_kwargs={"score_threshold": 0.5}, )

response = print(retriever.get_relevant_documents('What is Langchain?'))


query = "what is langchain?"
llm = ChatOllama(model='gemma:latest',temperatures=0)
# print(llm.invoke(query))


prompt = PromptTemplate.from_template(    """You are an AI language model assistant. 
Your task is to generate five different versions of the given user question to retrieve relevant documents from {context}. 


#ORIGINAL QUESTION: 
{question}
""")
chain = {"context":retriever,"question":RunnablePassthrough()} | prompt 
#print(chain)
print(chain.invoke(query))