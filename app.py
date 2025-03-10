import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('groq_API_KEY')

if not groq_api_key:
    st.error("Missing Groq API key. Please check your .env file.")
    st.stop()

# Initialize LLM
llm = ChatGroq(api_key=groq_api_key, model='deepseek-r1-distill-llama-70b')

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on the provided context only.\n"
               "Please provide the most accurate response based on the question.\n"
               "<context>\n"
               "{context}\n"
               "Question: {input}")
])

def create_vector_embedding():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        st.session_state.loader = PyPDFDirectoryLoader('Paper')  # Data ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        
        if not st.session_state.docs:
            st.error("No documents found in 'Paper' directory.")
            return
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        
        if st.session_state.final_documents:
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.success("Vector database is ready!")
        else:
            st.error("Document splitting failed. No final documents available.")

# Streamlit UI
st.title("Game of Thrones Document Retrieval System")

user_prompt = st.text_input("Enter your query from Game of Thrones")

if st.button("Create Document Embeddings"):
    create_vector_embedding()

if user_prompt:
    if 'vectors' not in st.session_state or st.session_state.vectors is None:
        st.error("Vector database is not initialized. Please run 'Create Document Embeddings' first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)
        
        start_time = time.process_time()
        response = retriever_chain.invoke({'input': user_prompt})
        elapsed_time = time.process_time() - start_time
        
        st.write("Response time:", elapsed_time)
        st.write(response.get('answer', 'No answer found.'))
        
        with st.expander("Document similarity search"):
            retrieved_docs = response.get('documents', [])
            for i, doc in enumerate(retrieved_docs):
                st.write(doc.page_content)
                st.write("-----------------------")
