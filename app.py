import re
import streamlit as st
from PyPDF2 import PdfReader
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Multilingual Citizen Chatbot", page_icon="💬", layout="wide")

st.title("💬 Multilingual Citizen Service Chatbot")
st.write("Upload a PDF, search a keyword/question, and get answers in English, Hindi, Telugu, Tamil, and Kannada.")

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

def translate_text(text, target):
    try:
        return GoogleTranslator(source="auto", target=target).translate(text)
    except:
        return text

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "vectors" not in st.session_state:
    st.session_state.vectors = None

with st.sidebar:
    st.header("📄 Upload PDF")
    pdf = st.file_uploader("Choose PDF file", type="pdf")

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

        st.sidebar.success("PDF processed successfully!")

keyword = st.chat_input("Search keyword or ask a question from the PDF...")

if keyword:
    if st.session_state.vectors is None:
        st.warning("Please upload a PDF first.")
    else:
        with st.chat_message("user"):
            st.write(keyword)

        keyword_en = translate_text(keyword, "en")

        q_vector = st.session_state.vectorizer.transform([keyword_en])
        scores = cosine_similarity(q_vector, st.session_state.vectors).flatten()

        top_indexes = scores.argsort()[::-1][:3]

        if scores[top_indexes[0]] < 0.01:
            answer = "No relevant answer found in the uploaded PDF."
        else:
            answer = " ".join([st.session_state.chunks[i] for i in top_indexes])

        with st.chat_message("assistant"):
            st.subheader("🌍 Answers in 5 Languages")

            st.markdown("### English")
            st.write(answer)

            st.markdown("### हिन्दी")
            st.write(translate_text(answer, "hi"))

            st.markdown("### తెలుగు")
            st.write(translate_text(answer, "te"))

            st.markdown("### தமிழ்")
            st.write(translate_text(answer, "ta"))

            st.markdown("### ಕನ್ನಡ")
            st.write(translate_text(answer, "kn"))

            with st.expander("📖 Source Chunks"):
                for i in top_indexes:
                    st.write(f"Score: {round(scores[i], 4)}")
                    st.write(st.session_state.chunks[i])
                    st.write("---")
