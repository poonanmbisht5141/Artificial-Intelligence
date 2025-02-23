#create streamlit application
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Streamlit App
st.title(" RAG Application: Chat with Your Documents")

# File uploader for PDF, DOCX, and TXT
uploaded_files = st.file_uploader(
    "Document.pdf)", 
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# Parse and extract text from uploaded documents
def parse_documents(files):
    text_data = ""
    for file in files:
        file_type = file.name.split(".")[-1].lower()
        
        if file_type == "pdf":
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text_data += page.extract_text() if page.extract_text() else ""

        elif file_type == "docx":
            doc = Document(file)
            for para in doc.paragraphs:
                text_data += para.text + "\n"

        elif file_type == "txt":
            text_data += file.read().decode("utf-8") + "\n"
    
    return text_data

# Process documents when files are uploaded
if uploaded_files:
    raw_text = parse_documents(uploaded_files)
    st.write("Documents Processed Successfully!")

    # Split text into chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([raw_text])

    # Initialize Embeddings and VectorStore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)

    # Set up the RetrievalQA Chain
    retriever = vector_store.as_retriever()
    llm = OpenAI(temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    # Chat Interface
    st.header(" Chat with Your Documents")
    query = st.text_input("Ask a question about the documents")

    if query:
        response = qa_chain.run(query)
        st.write(f" **Response:** {response}")


