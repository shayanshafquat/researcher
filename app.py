import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("RAG-Based Application")

# Use session state to store the current index_path, chat history, and summary
if 'current_index_path' not in st.session_state:
    st.session_state.current_index_path = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_summary' not in st.session_state:
    st.session_state.document_summary = None

# Function to get summary (with caching)
@st.cache_data
def get_document_summary(index_path):
    response = requests.post(f"{API_URL}/documents/summarize", json={"index_path": index_path})
    if response.status_code == 200:
        return response.json().get("summary", "No summary available.")
    else:
        return "Failed to summarize document."

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
                st.session_state.document_summary = None  # Reset summary when new document is uploaded
            else:
                st.error(f"Failed to upload {uploaded_file.name}")
else:
    document_link = st.text_input("Enter document link (including arXiv links):")
    if st.button("Process Link"):
        response = requests.post(f"{API_URL}/documents/process-link", json={"document_link": document_link})
        if response.status_code == 200:
            st.success(f"Processed document from link: {response.json().get('filename')}")
            st.session_state.current_index_path = response.json().get('index_path')
            st.session_state.document_summary = None  # Reset summary when new document is processed
        else:
            st.error("Failed to process document link")

# Summarize document
if st.session_state.current_index_path:
    if st.button("Summarize Document") or st.session_state.document_summary:
        if not st.session_state.document_summary:
            st.session_state.document_summary = get_document_summary(st.session_state.current_index_path)
        
        st.write("## Document Summary")
        st.markdown(st.session_state.document_summary)

# Chat interface
st.header("Chat with the Document")
user_input = st.text_input("You:", key="user_input")

if st.button("Send"):
    if st.session_state.current_index_path:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        response = requests.post(f"{API_URL}/documents/ask", json={"query": user_input, "index_path": st.session_state.current_index_path})
        if response.status_code == 200:
            answer = response.json().get("answer", "No answer found.")
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        else:
            st.error("Failed to get an answer.")
    else:
        st.error("Please upload a document or process a link first.")

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.write("You:", message["content"])
    else:
        st.write("Assistant:", message["content"])
