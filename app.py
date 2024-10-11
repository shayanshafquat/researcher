import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("RAG-Based Application")

# Upload documents
st.header("Upload Documents")
uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)

if st.button("Upload"):
    for uploaded_file in uploaded_files:
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post(f"{API_URL}/documents/upload", files=files)
        if response.status_code == 200:
            st.success(f"Uploaded {uploaded_file.name}")
        else:
            st.error(f"Failed to upload {uploaded_file.name}")

# Ask a question
st.header("Ask a Question")
query = st.text_input("Enter your question:")

if st.button("Submit"):
    response = requests.post(f"{API_URL}/documents/ask", json={"query": query})
    if response.status_code == 200:
        answer = response.json().get("answer", "No answer found.")
        st.write(f"Answer: {answer}")
    else:
        st.error("Failed to get an answer.")