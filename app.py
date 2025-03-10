

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()
## Load the groq api
os.environ['groq_API_KEY'] = os.getenv('groq_API_KEY')


groq_api_key = os.getenv('groq_API_KEY')

llm = ChatGroq(api_key=groq_api_key,model='deepseek-r1-distill-llama-70b')


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
        st.session_state.loader = PyPDFDirectoryLoader('Paper') ## this is my data ingestion step
        st.session_state.docs = st.session_state.loader.load() # document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap=300)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)



user_prompt = st.text_input('enter your query from the research paper ')

if st.button('Document Embedding'):
    create_vector_embedding()
    st.write('vector  database in ready')
    
    
import time 

if user_prompt:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever,document_chain)
    
    
    start = time.process_time()
    response = retriever_chain.invoke({'input':user_prompt})
    
    print(f'response time : {time.process_time()-start}')
    
    st.write(response['answer'])
    
    ## with streamplit expander
    
# Show document similarity search results
    with st.expander("Document similarity search"):
        retrieved_docs = response.get('documents', [])
        for i, doc in enumerate(retrieved_docs):
            st.write(doc.page_content)
            st.write("-----------------------")