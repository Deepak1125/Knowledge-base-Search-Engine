import os
import hashlib
import torch
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_faiss_retriever(pdf_path: str):
    # Create directory to store all FAISS indexes
    index_dir = "faiss_index"
    os.makedirs(index_dir, exist_ok=True)

    # Unique hash for the file
    with open(pdf_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    index_path = os.path.join(index_dir, file_hash)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={"device": device}
    )

    # Load and split PDF into chunks
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = os.path.basename(pdf_path)

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(documents)

    new_store = FAISS.from_documents(chunks, embeddings)
    new_store.save_local(index_path)

    if "vector_store" in st.session_state and st.session_state.vector_store is not None:
        st.session_state.vector_store.merge_from(new_store)
    else:
        st.session_state.vector_store = new_store

    return st.session_state.vector_store.as_retriever(search_kwargs={"k": 15})