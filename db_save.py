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
# documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
    )


pages = loader.load_and_split(text_splitter)

embedding_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-mpnet-base-v2',
                                        model_kwargs={'device':'cuda'},
                                        encode_kwargs={'normalize_embeddings':True})
doc_func = lambda x: x.page_content
docs = list(map(doc_func, pages))
vectorstore = FAISS.from_texts(docs,
                embedding=embedding_model,
                distance_strategy=DistanceStrategy.COSINE)
vectorstore.save_local('./db')


# pages = loader.load()

# print(pages)