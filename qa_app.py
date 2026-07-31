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

from langchain_classic.chains import RetrievalQA
from retriever import get_faiss_retriever
from llm import GeminiLLM

# -------------------- Style --------------------

def apply_custom_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
    }

    /* ---------- Global dark, glossy background ---------- */
    .stApp {
        background:
            radial-gradient(1200px 600px at 15% -10%, rgba(90,130,255,0.18), transparent 60%),
            radial-gradient(1000px 500px at 110% 10%, rgba(160,90,255,0.14), transparent 55%),
            linear-gradient(180deg, #0b0c10 0%, #101218 45%, #0b0c10 100%);
        color: #e8eaf0;
    }

    /* Hide default streamlit chrome clutter */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    #MainMenu, footer {visibility: hidden;}

    .app-container {
        max-width: 850px;
        margin: 0 auto;
        padding: 0 1rem;
    }

    /* ---------- Title ---------- */
    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 0%, #a9b6ff 55%, #7ea8ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding-bottom: 0.2rem;
        font-size: 2rem !important;
    }

    h3, h2 {
        color: #f2f3f7 !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }

    /* ---------- Glassy panels ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ---------- Chat bubbles: glossy Apple iMessage feel ---------- */
    .chat-container {
        display: inline-block;
        padding: 12px 16px;
        margin: 6px 0;
        border-radius: 18px;
        font-size: 15.5px;
        line-height: 1.5;
        max-width: 78%;
        word-wrap: break-word;
        box-shadow: 0 4px 18px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.15);
        position: relative;
    }

    .user-msg {
        background: linear-gradient(135deg, #3a82ff 0%, #1c64f2 55%, #0e4fd1 100%);
        color: white;
        margin-right: auto;
        margin-left: 0;
        text-align: left;
        border-bottom-left-radius: 6px;
    }

    .assistant-msg {
        background: linear-gradient(135deg, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.10);
        color: #eef0f5;
        margin-left: auto;
        margin-right: 0;
        display: block;
        text-align: left;
        border-bottom-right-radius: 6px;
    }

    .chat-row {
        display: flex;
        width: 100%;
    }
    .chat-row.user { justify-content: flex-start; }
    .chat-row.assistant { justify-content: flex-end; }

    .chat-scroll {
        max-height: 62vh;
        overflow-y: auto;
        padding: 8px 6px 8px 2px;
        display: flex;
        flex-direction: column;
    }

    .chat-scroll::-webkit-scrollbar {
        width: 8px;
    }
    .chat-scroll::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.15);
        border-radius: 10px;
    }
    .chat-scroll::-webkit-scrollbar-track {
        background: transparent;
    }

    /* ---------- Buttons: glossy pill ---------- */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(180deg, rgba(255,255,255,0.14), rgba(255,255,255,0.04));
        color: #f2f3f7;
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 12px;
        padding: 0.55rem 1rem;
        font-weight: 600;
        backdrop-filter: blur(10px);
        box-shadow: 0 2px 10px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.18);
        transition: all 0.15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: linear-gradient(180deg, rgba(90,130,255,0.35), rgba(90,130,255,0.15));
        border-color: rgba(120,150,255,0.5);
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(60,100,255,0.25), inset 0 1px 0 rgba(255,255,255,0.2);
    }
    .stButton > button:active, .stDownloadButton > button:active {
        transform: translateY(0px);
    }

    /* ---------- File uploader ---------- */
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1.5px dashed rgba(255,255,255,0.18);
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }

    /* ---------- Chat input ---------- */
    [data-testid="stChatInput"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    }

    /* ---------- Divider ---------- */
    hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* ---------- Alerts (info/success/error) glossy cards ---------- */
    div[data-testid="stAlert"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }

    /* ---------- Spinner text ---------- */
    .stSpinner > div {
        color: #cfd6ff;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------- Answer Formatter --------------------

def clean_and_format_answer(text: str) -> str:
    # Remove bullet noise and normalize spacing
    text = re.sub(r"[■]+", "", text)
    text = re.sub(r"\n\s*\n", "\n\n", text.strip())

    # Normalize bullets
    text = re.sub(r"\n\s*[\*\•\-]", "\n- ", text)

    # Remove triple backtick code blocks completely
    text = re.sub(r"```(?:.|\n)*?```", "", text, flags=re.DOTALL)

    # Remove indented lines (they render as code)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        if line.lstrip().startswith("```"):
            continue  # skip entire fenced code blocks
        if line.startswith("    ") or line.startswith("\t"):
            cleaned_lines.append(line.lstrip())  # strip indents to avoid block code
        else:
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # Replace inline block look with plain quotes
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
        # Clean up temp files
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
    st.title("TalkToPDF")
    st.caption("Implemented using RAG, LangChain and Gemini LLM")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "uploaded_file_names" not in st.session_state:
        st.session_state.uploaded_file_names = []

    with st.sidebar:
        st.markdown("## 📤 Upload PDFs")
        uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

        # Clear history button
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
        # Check if files have changed
        current_file_names = [f.name for f in uploaded_files]
        if current_file_names != st.session_state.uploaded_file_names:
            try:
                with st.spinner("🔍 Processing PDFs..."):
                    retriever = build_combined_retriever(uploaded_files)
                    llm = GeminiLLM(api_key=st.secrets["GEMINI_API_KEY"])
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
                    st.session_state.uploaded_file_names = current_file_names
                    # Clear history when new PDFs are uploaded
                    st.session_state.chat_history = []
                st.success("✅ PDFs processed successfully!")
            except Exception as e:
                st.error(f"❌ Error processing PDFs: {str(e)}")
                return

        if st.session_state.qa_chain:
            st.divider()
            st.markdown("### 💬 Ask questions from your PDFs")

            # Display chat history
            chat_area = st.container()
            with chat_area:
                st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
                for message in st.session_state.chat_history:
                    st.markdown(
                        f'<div class="chat-row user"><div class="chat-container user-msg">{message["question"]}</div></div>',
                        unsafe_allow_html=True
                    )
                    formatted_answer = clean_and_format_answer(message["answer"])
                    st.markdown(
                        f'<div class="chat-row assistant"><div class="chat-container assistant-msg">{formatted_answer}</div></div>',
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)

            # Chat input
            user_input = st.chat_input("Type your question here...")

            if user_input:
                # Build context from last 3 exchanges only (to avoid token limit)
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
                    
                    # Rerun to display new message
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
    else:
        st.info("👈 Please upload PDF files to get started")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Run --------------------
if __name__ == "__main__":
    run_qa_app()