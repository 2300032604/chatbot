import re
import streamlit as st
from PyPDF2 import PdfReader
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Multilingual Citizen Service Chatbot",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Multilingual Citizen Service Chatbot")
st.write("Upload a PDF and ask questions from the document.")

def extract_text(pdf):
    reader = PdfReader(pdf)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    return re.sub(r"\s+", " ", text).strip()

def create_chunks(text, chunk_size=800):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) > 100:
            chunks.append(chunk)

    return chunks

def translate_to_english(question):
    try:
        return GoogleTranslator(source="auto", target="en").translate(question)
    except:
        return question

def translate_answer(answer, lang):
    if lang == "English":
        return answer

    code = {
        "Telugu": "te",
        "Hindi": "hi",
        "Tamil": "ta",
        "Kannada": "kn"
    }[lang]

    try:
        return GoogleTranslator(source="en", target=code).translate(answer)
    except:
        return answer

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "vectors" not in st.session_state:
    st.session_state.vectors = None

with st.sidebar:
    st.header("📄 Upload Dataset")
    pdf = st.file_uploader("Upload PDF", type="pdf")

    language = st.selectbox(
        "Answer Language",
        ["English", "Telugu", "Hindi", "Tamil", "Kannada"]
    )

if pdf:
    if "processed_pdf" not in st.session_state or st.session_state.processed_pdf != pdf.name:
        with st.spinner("Reading and indexing PDF..."):
            text = extract_text(pdf)

            if not text:
                st.error("No readable text found in PDF.")
                st.stop()

            chunks = create_chunks(text)

            vectorizer = TfidfVectorizer(stop_words="english")
            vectors = vectorizer.fit_transform(chunks)

            st.session_state.chunks = chunks
            st.session_state.vectorizer = vectorizer
            st.session_state.vectors = vectors
            st.session_state.processed_pdf = pdf.name
            st.session_state.messages = []

        st.sidebar.success("PDF processed successfully!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask anything from your uploaded document...")

if question:
    if st.session_state.vectors is None:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        question_en = translate_to_english(question)

        q_vector = st.session_state.vectorizer.transform([question_en])
        scores = cosine_similarity(q_vector, st.session_state.vectors).flatten()

        top_indexes = scores.argsort()[::-1][:3]

        if scores[top_indexes[0]] < 0.01:
            answer = "No relevant answer found in the uploaded document."
            sources = []
        else:
            answer = " ".join([st.session_state.chunks[i] for i in top_indexes])
            answer = translate_answer(answer, language)
            sources = top_indexes

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.write(answer)

            if len(sources) > 0:
                with st.expander("📖 Source Chunks"):
                    for i in sources:
                        st.write(f"Score: {round(scores[i], 4)}")
                        st.write(st.session_state.chunks[i])
                        st.write("---")
