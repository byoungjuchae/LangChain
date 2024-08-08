from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
import os

os.environ["LANGCHAIN_TRAICING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]  = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]  = "agent1"
os.environ["TAVILY_API_KEY"] = ""


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_siz=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                  model_kwargs={'device':'cuda'})
vectorstore = FAISS.from_documents(
    documents =  doc_splits,
    embedding=embedidng
)

llm = ChatOllama(model='llama3.1:latest',format='json',temperature=0)
llm_grade = llm.with_structured_output()
retriever = vectorstore.as_retriever()
