import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from vectorstore import add_documents_to_vectorstore

def process_all_pdfs(uploaded_files, vectorstore):
    docs = []

    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_file_path = tmp_file.name

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        pdf_docs = loader.load()

        # Split the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        pdf_docs = text_splitter.split_documents(pdf_docs)
        docs.extend(pdf_docs)

        # Optionally, delete the temporary file
        os.remove(temp_file_path)

    # Add all documents to the vector store at once
    add_documents_to_vectorstore(docs, vectorstore)
