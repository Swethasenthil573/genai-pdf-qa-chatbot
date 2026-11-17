## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
AI-Powered PDF Question Answering Chatbot Students often spend a lot of time searching through lengthy PDF documents to find specific information. Reading entire documents to identify relevant answers can be time-consuming and difficult.

The proposed system aims to develop an AI-powered PDF Question Answering Chatbot using LangChain, OpenAI, and ChromaDB. Users can upload a PDF, ask questions in natural language, and receive relevant answers based on the document content.
### DESIGN STEPS:
1.Upload the PDF document using PyPDFLoader.

2.Extract and split the text into smaller chunks.

3.Convert text chunks into embeddings and store them in ChromaDB.

4.Retrieve relevant information based on the user's question.

5.Generate and display answers using the OpenAI language model.

### PROGRAM:
```
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # Load OpenAI API key
openai_api_key = os.environ['OPENAI_API_KEY']

def load_pdf_to_db(file_path):
    # Load the PDF file
    loader = PyPDFLoader("Machine Learning By Thiru.pdf")
    documents = loader.load()
    
    # Split the documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    
    # Embed the documents
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    
    # Set retriever to fetch relevant document chunks
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever
def create_conversational_chain(retriever):
    # Initialize memory for conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Define the conversational retrieval chain
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),  # Using OpenAI Chat model
        retriever=retriever,
        memory=memory
    )
    return conversational_chain
if __name__ == "__main__":
    # Load the PDF file
    pdf_file_path = "Machine Learning By Thiru.pdf"  # Replace with your file path
    retriever = load_pdf_to_db(pdf_file_path)
    
    # Create chatbot chain
    chatbot = create_conversational_chain(retriever)
    
    # Start a conversation
    print("Welcome to the PDF Question-Answering Chatbot!")
    print("Type 'exit' to quit.")
    
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Chatbot: Thanks for chatting! Goodbye!")
            break
        
        result = chatbot({"question": user_query})
        print("Chatbot:", result["answer"])
```
### OUTPUT:
<img width="1600" height="672" alt="AI exp 3" src="https://github.com/user-attachments/assets/0219bf2e-6991-4f48-b6b2-8c6aa16ac951" />
### RESULT:
The AI-powered PDF Question Answering Chatbot was successfully developed using LangChain, OpenAI, and ChromaDB. It extracts text from PDF documents, stores the text as embeddings, and retrieves relevant information based on user questions. The chatbot generates answers in natural language, making information retrieval faster, easier, and more efficient.
