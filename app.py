import streamlit as st
import requests
from config.model_config import ModelProvider
from pathlib import Path

API_URL = "http://localhost:8000"

# Load CSS
def load_css():
    css_file = Path(__file__).parent / "static" / "styles.css"
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="ResearchGPT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS styles
load_css()

# Application Title
st.markdown('<h1 class="app-title">ResearchGPT</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Your AI Research Assistant for Scientific Papers</p>', unsafe_allow_html=True)

# Model selection in sidebar
st.sidebar.header("Model Settings")
model_provider = st.sidebar.selectbox(
    "Choose Model Provider",
    options=[ModelProvider.OPENAI, ModelProvider.LOCAL],
    format_func=lambda x: "OpenAI API" if x == ModelProvider.OPENAI else "Local LLM (Mistral)"
)

# Initialize session state
if 'current_index_path' not in st.session_state:
    st.session_state.current_index_path = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_summary' not in st.session_state:
    st.session_state.document_summary = None
if 'model_provider' not in st.session_state:
    st.session_state.model_provider = model_provider
if 'send_message' not in st.session_state:
    st.session_state.send_message = False

# Update model provider if changed
if st.session_state.model_provider != model_provider:
    st.session_state.model_provider = model_provider
    st.session_state.chat_history = []
    st.session_state.document_summary = None

@st.cache_data
def get_document_summary(index_path, model_provider):
    response = requests.post(
        f"{API_URL}/documents/summarize", 
        json={"index_path": index_path, "model_provider": model_provider}
    )
    if response.status_code == 200:
        return response.json().get("summary", "No summary available.")
    else:
        return "Failed to summarize document."

# Callback for sending messages
def send_message():
    st.session_state.send_message = True

# Upload documents or process links
st.header("Upload Documents or Process Links")
upload_option = st.radio("Choose an option:", ("Upload Files", "Paste Document Link"))

if upload_option == "Upload Files":
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
    if st.button("Upload"):
        for uploaded_file in uploaded_files:
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(f"{API_URL}/documents/upload", files=files)
            if response.status_code == 200:
                st.success(f"Uploaded and processed: {uploaded_file.name}")
                st.session_state.current_index_path = response.json().get('index_path')
                st.session_state.document_summary = None
            else:
                st.error(f"Failed to upload {uploaded_file.name}")
else:
    document_link = st.text_input("Enter document link (including arXiv links):")
    if st.button("Process Link"):
        response = requests.post(f"{API_URL}/documents/process-link", json={"document_link": document_link})
        if response.status_code == 200:
            st.success(f"Processed document from link: {response.json().get('filename')}")
            st.session_state.current_index_path = response.json().get('index_path')
            st.session_state.document_summary = None
        else:
            st.error("Failed to process document link")

# Summarize document
if st.session_state.current_index_path:
    if st.button("Summarize Document") or st.session_state.document_summary:
        if not st.session_state.document_summary:
            st.session_state.document_summary = get_document_summary(
                st.session_state.current_index_path,
                st.session_state.model_provider
            )
        
        st.markdown("---")
        st.markdown("## 📄 Document Summary")
        st.markdown(f"""
        <div class="summary-container">
            {st.session_state.document_summary}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

# Chat interface
st.header("💬 Chat with the Document")

# Display chat history
if st.session_state.chat_history:
    st.subheader("Conversation History")
    for message in st.session_state.chat_history:
        message_class = "user-message" if message["role"] == "user" else "assistant-message"
        st.markdown(f'<div class="{message_class}">{message["content"]}</div>', unsafe_allow_html=True)

# Input for new message with callback
user_input = st.text_input(
    "Ask a question about the document:",
    key="user_input",
    placeholder="Example: What are the main findings of this research?",
    on_change=send_message
)

# Handle message sending
if st.session_state.send_message:
    if st.session_state.current_index_path and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        response = requests.post(
            f"{API_URL}/documents/ask",
            json={
                "query": user_input,
                "index_path": st.session_state.current_index_path,
                "model_provider": st.session_state.model_provider
            }
        )
        
        if response.status_code == 200:
            answer = response.json().get("answer", "No answer found.")
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        else:
            st.error("Failed to get an answer.")
    
    st.session_state.send_message = False
    st.rerun()