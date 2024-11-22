import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

# Load the GROQ API Key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# If you do not have open AI key use the below Huggingface embedding
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

llm = Ollama(model="llama3.2")


prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)


def create_vector_store(documents):
    try:
        if "vectorstore" not in st.session_state:
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Ensure the loader is initialized correctly
            st.session_state.loader = PyPDFDirectoryLoader(os.path.dirname("temp.pdf"))
            st.session_state.documents = st.session_state.loader.load()
            
            if not st.session_state.documents:
                raise ValueError("No documents found in the specified directory")
                
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            st.session_state.texts = st.session_state.text_splitter.split_documents(
                st.session_state.documents
            )
            st.session_state.vectorstore = FAISS.from_documents(
                st.session_state.texts, 
                st.session_state.embeddings
            )
            return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

# UI Components
st.title("Document Q&A Bot")

# File uploader for PDF documents
uploaded_files = st.file_uploader("Upload multiple PDF documents", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load the PDF and create vector store
        documents = [uploaded_file]  # Directly use the uploaded file
        if create_vector_store(documents):
            st.success("Document Embedding Complete")

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_prompt = st.text_input("Type your question:")

if st.button("Send"):
    if not st.session_state.vectorstore:
        st.error("Please upload a document first.")
    else:
        try:
            with st.spinner("Processing your question..."):
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectorstore.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                start_time = time.time()
                response = retrieval_chain.invoke({"input": user_prompt})
                processing_time = time.time() - start_time
                
                # Display user message
                st.session_state.chat_history.append({"role": "user", "content": user_prompt})
                st.markdown(f"<div class='user-message'>{user_prompt}</div>", unsafe_allow_html=True)

                # Display bot response
                chatbot_response = response["answer"]
                st.session_state.chat_history.append({"role": "bot", "content": chatbot_response})
                st.markdown(f"<div class='bot-message'>{chatbot_response}</div>", unsafe_allow_html=True)

                st.info(f"Processing time: {processing_time:.2f} seconds")

        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

# Render chat history
st.markdown("### Chat History")
for message in st.session_state.chat_history:
    css_class = "user-message" if message["role"] == "user" else "bot-message"
    st.markdown(f"<div class='{css_class}'>{message['content']}</div>", unsafe_allow_html=True)

# Add a clear button to reset the session state
if st.button("Clear Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Session cleared successfully!")

# Add CSS for custom styles
st.markdown(
    """
    <style>
        body {
            background-color: white;  /* Change background color to white */
            color: black;  /* Set text color to black for better visibility */
        }
        .user-message {
            background-color: #133E87;  /* Light green for user messages */
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            text-align: right;
            float: right;
            clear: both;
            max-width: 70%;
        }
        .bot-message {
            background-color: #FF3F3E0;  /* Light gray for bot messages */
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            text-align: left;
            float: left;
            clear: both;
            max-width: 70%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)