import streamlit as st
from rag_system import generate_rag_prompt, get_relevant_context_from_db, generate_answer
from pdf_processing import process_all_pdfs
from vectorstore import initialize_vectorstore


def main():
    # Initialize session state for vectorstore and uploaded PDFs
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = initialize_vectorstore()
    if 'uploaded_pdfs' not in st.session_state:
        st.session_state.uploaded_pdfs = []

    st.title("RAG System with PDF Upload")

    # PDF Upload Section
    st.write("Upload PDF files to add new information to the database:")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        st.session_state.uploaded_pdfs.extend(uploaded_files)
        st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully.")

    # Button to start vectorizing all uploaded PDFs
    if st.button("Start Vectorizing PDFs"):
        if st.session_state.uploaded_pdfs:
            process_all_pdfs(st.session_state.uploaded_pdfs, st.session_state.vectorstore)
            st.success("All PDFs have been vectorized and the database has been updated successfully.")
        else:
            st.warning("No PDFs uploaded yet. Please upload PDFs before vectorizing.")

    st.write("-----------------------------------------------------------------------\n")

    # Chat Box Section
    st.write("Ask me anything and I will provide answers based on the context from my database.")
    query = st.text_input("Your Question:")
    if st.button("Get Answer"):
        if query:
            context = get_relevant_context_from_db(query)
            prompt = generate_rag_prompt(query=query, context=context)
            answer = generate_answer(prompt=prompt)
            st.write(answer)
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
