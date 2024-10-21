import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("RAG-Based Application")

# Use session state to store the current index_path
if 'current_index_path' not in st.session_state:
    st.session_state.current_index_path = None

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
            else:
                st.error(f"Failed to upload {uploaded_file.name}")
else:
    document_link = st.text_input("Enter document link (including arXiv links):")
    if st.button("Process Link"):
        response = requests.post(f"{API_URL}/documents/process-link", json={"document_link": document_link})
        if response.status_code == 200:
            st.success(f"Processed document from link: {response.json().get('filename')}")
            st.session_state.current_index_path = response.json().get('index_path')
        else:
            st.error("Failed to process document link")

# Ask a question
st.header("Ask a Question")
query = st.text_input("Enter your question:")

if st.button("Submit"):
    if st.session_state.current_index_path:
        response = requests.post(f"{API_URL}/documents/ask", json={"query": query, "index_path": st.session_state.current_index_path})
        if response.status_code == 200:
            answer = response.json().get("answer", "No answer found.")
            st.write(f"Answer: {answer}")
        else:
            st.error("Failed to get an answer.")
    else:
        st.error("Please upload a document or process a link first.")
