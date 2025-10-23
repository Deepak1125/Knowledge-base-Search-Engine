from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import streamlit as st  # to access secrets
from retriever import build_retriever
from llm import ask_llm

# Load API key from secrets
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY  # optional if your llm code reads from env

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_file(file: UploadFile):
    temp_path = f"./uploads/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    build_retriever(temp_path)
    return {"message": f"File {file.filename} processed"}

@app.post("/ask")
async def ask_question(query: str = Form(...)):
    answer = ask_llm(query)
    return {"answer": answer}
