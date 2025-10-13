import os
import hashlib
import torch
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_faiss_retriever(pdf_path: str):
    index_dir = "faiss_index"
    os.makedirs(index_dir, exist_ok=True)

    with open(pdf_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    index_path = os.path.join(index_dir, file_hash)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={"device": device}
    )

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = os.path.basename(pdf_path)

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(documents)

    # Assign unique IDs to avoid collisions
    from uuid import uuid4
    for chunk in chunks:
        chunk.metadata["id"] = str(uuid4())

    new_store = FAISS.from_documents(chunks, embeddings)
    new_store.save_local(index_path)

    # Merge or create session vector store
    if "vector_store" in st.session_state and st.session_state.vector_store is not None:
        st.session_state.vector_store.merge_from(new_store)
    else:
        st.session_state.vector_store = new_store

    return st.session_state.vector_store.as_retriever(search_kwargs={"k": 15})
