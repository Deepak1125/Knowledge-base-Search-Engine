import streamlit as st
import tempfile
import os
import json
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import inch
import re

from langchain.chains import RetrievalQA
from retriever import get_faiss_retriever
from llm import GeminiLLM

# -------------------- Style --------------------

def apply_custom_style():
    st.markdown("""
    <style>
    .app-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    .chat-container {
        display: inline-block;
        padding: 10px 14px;
        margin: 4px 0;
        border-radius: 10px;
        font-size: 16px;
        line-height: 1.4;
        max-width: 80%;
        word-wrap: break-word;
    }
    .user-msg {
        background-color: #1c64f2;
        color: white;
        margin-right: auto;
        text-align: left;
    }
    .assistant-msg {
        background-color: #f1f1f1;
        color: black;
        margin-left: auto;
    }
    .chat-scroll {
        max-height: 60vh;
        overflow-y: auto;
        padding-right: 10px;
        display: flex;
        flex-direction: column;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------- Answer Formatter --------------------

def clean_and_format_answer(text: str) -> str:
    text = re.sub(r"[■]+", "", text)
    text = re.sub(r"\n\s*\n", "\n\n", text.strip())
    text = re.sub(r"\n\s*[\*\•\-]", "\n- ", text)
    text = re.sub(r"```(?:.|\n)*?```", "", text, flags=re.DOTALL)

    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        if line.lstrip().startswith("```"):
            continue  
        if line.startswith("    ") or line.startswith("\t"):
            cleaned_lines.append(line.lstrip())  
        else:
            cleaned_lines.append(line)
            
    text = "\n".join(cleaned_lines)
    text = re.sub(r"`([^`]+)`", r'"\1"', text)

    return text.strip()
    
# -------------------- PDF Generator --------------------

def generate_chat_pdf(chat_history):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=40, leftMargin=40,
                            topMargin=60, bottomMargin=40)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Question", fontSize=12, leading=16, spaceAfter=6,
                              textColor="#1c64f2", fontName="Helvetica-Bold", alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="Answer", fontSize=12, leading=16, spaceAfter=12,
                              fontName="Helvetica", alignment=TA_LEFT))

    flowables = []

    for i, msg in enumerate(chat_history, start=1):
        q_text = f"<b>Q{i}:</b> {msg['question']}"
        a_text = f"<b>A{i}:</b> {clean_and_format_answer(msg['answer'])}"
        flowables.append(Paragraph(q_text, styles["Question"]))
        flowables.append(Paragraph(a_text, styles["Answer"]))
        flowables.append(Spacer(1, 0.2 * inch))

    doc.build(flowables)
    buffer.seek(0)
    return buffer

# -------------------- Retriever --------------------
def build_combined_retriever(uploaded_files):
    temp_paths = []
    try:
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                temp_paths.append(tmp.name)

        retrievers = [get_faiss_retriever(path) for path in temp_paths]
        base_retriever = retrievers[0]
        for r in retrievers[1:]:
            base_retriever.vectorstore.merge_from(r.vectorstore)

        return base_retriever
    finally:
        for path in temp_paths:
            try:
                os.unlink(path)
            except:
                pass

# -------------------- App --------------------

def run_qa_app():
    st.set_page_config(page_title="Gemini PDF Chat", page_icon="🤖", layout="wide")
    apply_custom_style()

    st.markdown('<div class="app-container">', unsafe_allow_html=True)
    st.title("Knowledge Base Search Engine  Implemented using RAG, LangChain and Gemini LLM")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "uploaded_file_names" not in st.session_state:
        st.session_state.uploaded_file_names = []

    with st.sidebar:
        st.markdown("## 📤 Upload PDFs")
        uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("## 💾 Download Chat History")

            history_text = "\n".join([f"Q: {m['question']}\nA: {clean_and_format_answer(m['answer'])}\n" for m in st.session_state.chat_history])
            history_json = json.dumps(st.session_state.chat_history, indent=2)
            history_pdf = generate_chat_pdf(st.session_state.chat_history)

            st.download_button("📄 Download as TXT", history_text, "chat_history.txt", "text/plain")
            st.download_button("🧾 Download as JSON", history_json, "chat_history.json", "application/json")
            st.download_button("📥 Download as PDF", history_pdf, file_name="chat_history.pdf", mime="application/pdf")

    if uploaded_files:
        current_file_names = [f.name for f in uploaded_files]
        if current_file_names != st.session_state.uploaded_file_names:
            try:
                with st.spinner("🔍 Processing PDFs..."):
                    retriever = build_combined_retriever(uploaded_files)
                    llm = GeminiLLM(api_key=st.secrets["GEMINI_API_KEY"])
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
                    st.session_state.uploaded_file_names = current_file_names
                    st.session_state.chat_history = []
                st.success("✅ PDFs processed successfully!")
            except Exception as e:
                st.error(f"❌ Error processing PDFs: {str(e)}")
                return

        if st.session_state.qa_chain:
            st.divider()
            st.markdown("### 💬 Ask questions from your PDFs")

            chat_area = st.container()
            with chat_area:
                for message in st.session_state.chat_history:
                    st.markdown(f'<div class="chat-container user-msg">{message["question"]}</div>', unsafe_allow_html=True)
                    formatted_answer = clean_and_format_answer(message["answer"])
                    st.markdown(f'<div class="chat-container assistant-msg">{formatted_answer}</div>', unsafe_allow_html=True)

            user_input = st.chat_input("Type your question here...")

            if user_input:
                recent_history = st.session_state.chat_history[-3:] if len(st.session_state.chat_history) > 3 else st.session_state.chat_history
                context_prompt = "\n".join([
                    f"You: {m['question']}\nGemini: {m['answer']}"
                    for m in recent_history
                ])
                full_prompt = f"{context_prompt}\nYou: {user_input}\nGemini:" if context_prompt else f"You: {user_input}\nGemini:"
                
                try:
                    with st.spinner("🤖 Thinking..."):
                        raw_answer = st.session_state.qa_chain.run(full_prompt)
                    
                    # Add to history
                    st.session_state.chat_history.append({
                        "question": user_input,
                        "answer": raw_answer
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
    else:
        st.info("👈 Please upload PDF files to get started")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Run --------------------
if __name__ == "__main__":
    run_qa_app()
