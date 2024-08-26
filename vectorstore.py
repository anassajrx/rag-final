from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def initialize_vectorstore():
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    # Initialize Chroma vector store with a persistent directory
    vectorstore = Chroma(
        persist_directory="./chroma_db_nccn",  # Directory where the vector store is saved
        embedding_function=embedding_function
    )
    return vectorstore

def add_documents_to_vectorstore(docs, vectorstore):
    # Add documents to the vector store
    vectorstore.add_documents(docs)
    # The vector store persists automatically when initialized with persist_directory
