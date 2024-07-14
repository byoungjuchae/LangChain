from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS  
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

llm = Ollama(model="gemma:latest")
# response = llm.invoke("지구의 자전주기는?")W

loader = PyPDFDirectoryLoader('./folder')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
    )


data = text_splitter.split_text(documents[0].page_content)

embedding_model = HuggingFaceEmbeddings(model_name = 'jhgan/ko-sbert-nli',
                                        model_kwargs={'device':'cuda'},
                                        encode_kwargs={'normalize_embeddings':True})

vectorstore = FAISS.from_texts(data,
                embedding=embedding_model,
                distance_strategy=DistanceStrategy.COSINE)
vectorstore.save_local('./db')


pages = loader.load(text_splitter)

print(pages)