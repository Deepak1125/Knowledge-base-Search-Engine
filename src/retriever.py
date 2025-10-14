import os
import hashlib
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

def get_faiss_retriever(pdf_path: str):
    with open(pdf_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    index_dir = "faiss_index"
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, file_hash)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use HuggingFace embeddings with E5-base-v2
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={"device": device}
    )

    if os.path.exists(os.path.join(index_path, "index.faiss")):
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata["source"] = os.path.basename(pdf_path)

        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        chunks = splitter.split_documents(documents)
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(index_path)

    return vector_store.as_retriever(search_kwargs={"k": 15})
