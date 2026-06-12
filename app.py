import re
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
import faiss
import numpy as np

st.set_page_config(
    page_title="Multilingual Citizen Service Chatbot",
    page_icon="💬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #212121;
    color: white;
}
.chat-title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    margin-top: 20px;
}
.chat-subtitle {
    text-align: center;
    color: #b4b4b4;
    margin-bottom: 30px;
}
.user-msg {
    background: #2f2f2f;
    padding: 15px;
    border-radius: 18px;
    margin: 10px 0 10px auto;
    max-width: 70%;
}
.bot-msg {
    background: #303030;
    padding: 15px;
    border-radius: 18px;
    margin: 10px auto 10px 0;
    max-width: 75%;
}
.source-box {
    background: #171717;
    border-left: 4px solid #10a37f;
    padding: 12px;
    border-radius: 10px;
    margin-top: 10px;
    color: #d1d5db;
}
.stTextInput input {
    background-color: #303030;
    color: white;
    border-radius: 25px;
    padding: 15px;
    border: 1px solid #555;
}
.stFileUploader {
    background: #2f2f2f;
    padding: 15px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='chat-title'>💬 Multilingual Citizen Service Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='chat-subtitle'>Upload an official PDF and ask questions like ChatGPT</div>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

with st.sidebar:
    st.header("📄 Upload Dataset")
    pdf = st.file_uploader("Upload PDF", type="pdf")

    st.markdown("---")
    st.write("### Supported Languages")
    st.write("English")
    st.write("Telugu")
    st.write("Hindi")
    st.write("Tamil")
    st.write("Kannada")

def process_pdf(pdf):
    text = ""

    reader = PdfReader(pdf)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    text = re.sub(r"\s+", " ", text).strip()

    chunks = []
    chunk_size = 800

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) > 100:
            chunks.append(chunk)

    embeddings = model.encode(chunks, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))

    return chunks, index

if pdf:
    if (
        "processed_pdf" not in st.session_state
        or st.session_state.processed_pdf != pdf.name
    ):
        with st.spinner("Reading and indexing PDF..."):
            chunks, index = process_pdf(pdf)

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.processed_pdf = pdf.name
            st.session_state.messages = []

        st.sidebar.success("PDF processed successfully!")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'><b>You:</b><br>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'><b>Assistant:</b><br>{msg['content']}</div>", unsafe_allow_html=True)

question = st.chat_input("Ask anything from your uploaded document...")

if question:
    if st.session_state.index is None:
        st.warning("Please upload a PDF dataset first.")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        st.markdown(f"<div class='user-msg'><b>You:</b><br>{question}</div>", unsafe_allow_html=True)

        try:
            question_en = GoogleTranslator(
                source="auto",
                target="en"
            ).translate(question)
        except:
            question_en = question

        q_embedding = model.encode(
            [question_en],
            convert_to_numpy=True
        )

        distances, indices = st.session_state.index.search(
            q_embedding.astype("float32"),
            3
        )

        retrieved_chunks = [
            st.session_state.chunks[i]
            for i in indices[0]
            if i < len(st.session_state.chunks)
        ]

        answer = retrieved_chunks[0]

        try:
            Telugu = GoogleTranslator(source="en", target="te").translate(answer)
            Hindi = GoogleTranslator(source="en", target="hi").translate(answer)
        except:
            Telugu = answer
            Hindi = answer

        final_answer = f"""
{answer}

<br><br><b>తెలుగు:</b><br>{Telugu}

<br><br><b>हिन्दी:</b><br>{Hindi}
"""

        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer
        })

        st.markdown(
            f"<div class='bot-msg'><b>Assistant:</b><br>{final_answer}</div>",
            unsafe_allow_html=True
        )

        with st.expander("📖 Retrieved Source Chunks"):
            for i, chunk in enumerate(retrieved_chunks, start=1):
                st.markdown(
                    f"""
                    <div class='source-box'>
                    <b>Source {i}</b><br><br>
                    {chunk}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
