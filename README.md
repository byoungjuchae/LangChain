This Repository includs the Langchain, Langgraph, RAG and Agent AI. 

- Langchain
    - simple_chatmodel : It is the file to use fastapi and llama3.1. If you ask the question and answer about it. 
    - pdf_RAG : It is the file to retrieve the pdf file and answer the question with llama3.1, It only uses one pdf file.
    - pdfolder_RAG : It is the file to retrieve the pdf file and answer the question with llama3.1, It only uses one pdf folder file.
    - WebSearch : It is the file to refer the WebAPI, Tavily. And then, you get the answer with llama3.1
    - WebSearch_RAG : It is the file to refer the WebAPI and retrieve the pdf file. 

- Langgraph
   - 
----------------------------------------------------------------------------------------------------------------------------


To do 

- [ ] practice the RAG method such as simpleRAG, SqlRAG. These are refered to the langgraph docs. 
- [ ] dockerize the upper folder.
- [ ] Make docs each folder. 
- [ ] Code Assistant. Refer to langgraph docs.
- [x] SR model as tool. 

08.18.2024

- [x] Use the Tool such as TavilyAPI.
- [x] Using Langgraph to make simple chat model and Web Surfing.
- [x] Using Langgraph and use checkpointer which is ability to memorize previous conversation.
