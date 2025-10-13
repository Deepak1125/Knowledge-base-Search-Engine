import os
import hashlib
import time
import torch
import streamlit as st
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_faiss_retriever(pdf_path: str, allow_duplicate_version: bool = False):
    index_dir = "faiss_index"
    os.makedirs(index_dir, exist_ok=True)

    # 🔐 Read file content for hashing
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    # 📅 Optional: Add timestamp to hash if duplicates are allowed
    if allow_duplicate_version:
        file_hash = hashlib.md5(file_bytes + str(time.time()).encode()).hexdigest()
    else:
        file_hash = hashlib.md5(file_bytes).hexdigest()

    index_path = os.path.join(index_dir, file_hash)

    # 🧠 If index already exists and duplicates not allowed, just load it
    if os.path.exists(index_path) and not allow_duplicate_version:
        print(f"[INFO] Index for '{os.path.basename(pdf_path)}' already exists. Skipping embedding.")
        existing_store = FAISS.load_local(
            index_path,
            HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2"),
            allow_dangerous_deserialization=True
        )
        # Merge into session store
        if "vector_store" in st.session_state and st.session_state.vector_store is not None:
            st.session_state.vector_store.merge_from(existing_store)
        else:
            st.session_state.vector_store = existing_store
        return st.session_state.vector_store.as_retriever(search_kwargs={"k": 15})

    # 🧠 Choose device for embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={"device": device}
    )

    # 📄 Load and split document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = os.path.basename(pdf_path)

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(documents)

    # 🪄 Assign unique chunk IDs
    for chunk in chunks:
        chunk.metadata["id"] = str(uuid4())

    # 🧭 Create FAISS index and save it
    new_store = FAISS.from_documents(chunks, embeddings)
    new_store.save_local(index_path)

    # 🧩 Merge or initialize session vector store
    if "vector_store" in st.session_state and st.session_state.vector_store is not None:
        st.session_state.vector_store.merge_from(new_store)
    else:
        st.session_state.vector_store = new_store

    print(f"[INFO] Successfully embedded: {os.path.basename(pdf_path)}")
    return st.session_state.vector_store.as_retriever(search_kwargs={"k": 15})
