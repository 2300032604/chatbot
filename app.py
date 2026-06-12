import streamlit as st
from PyPDF2 import PdfReader

st.set_page_config(
    page_title="S.NO-362 Citizen Service Chatbot",
    page_icon="🏛️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #111827);
    color: white;
}
.main-box {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 0 25px rgba(0,0,0,0.4);
}
.source-box {
    background: rgba(56,189,248,0.12);
    padding: 15px;
    border-radius: 12px;
    margin-top: 12px;
}
h1, h2, h3 {
    color: white;
}
.stTextInput input {
    border-radius: 12px;
}
.stFileUploader {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center;'>🏛️ Multilingual Citizen-Service Chatbot</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;font-size:18px;'>S.NO-362 | Ask questions from official documents</p>",
    unsafe_allow_html=True
)

translation_keywords = {
    "జనన ధృవీకరణ": "birth certificate",
    "పుట్టిన సర్టిఫికేట్": "birth certificate",
    "రేషన్ కార్డు": "ration card",
    "పెన్షన్": "pension",
    "ఫిర్యాదు": "grievance complaint",
    "आय प्रमाण पत्र": "income certificate",
    "राशन कार्ड": "ration card",
    "पेंशन": "pension",
    "शिकायत": "grievance complaint",
    "आधार": "aadhaar"
}

blocked_keywords = [
    "hack",
    "bypass",
    "fake document",
    "forge certificate",
    "delete records",
    "leak data",
    "admin password"
]

def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    return file.read().decode("utf-8", errors="ignore")


def normalize_query(question):
    q = question.lower()

    for key, value in translation_keywords.items():
        if key.lower() in q:
            q += " " + value

    return q


def guardrail_check(question):
    q = question.lower()

    for word in blocked_keywords:
        if word in q:
            return False

    return True


def search_answer(text, question):
    question = normalize_query(question)

    paragraphs = text.split("\n")
    words = question.lower().split()
    scored = []

    for para in paragraphs:
        score = sum(1 for w in words if w in para.lower())

        if score > 0 and len(para.strip()) > 20:
            scored.append((score, para))

    scored.sort(reverse=True)

    if not scored:
        return "No matching answer found in the uploaded official document.", []

    answer = "\n\n".join([p for s, p in scored[:5]])
    sources = scored[:5]

    return answer, sources


col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Official Document")
    uploaded_file = st.file_uploader(
        "Choose PDF or TXT file",
        type=["pdf", "txt"]
    )

with col2:
    st.markdown("### 🌐 Select Language")
    language = st.selectbox(
        "Response Language",
        ["English", "Hindi", "Telugu"]
    )

    st.markdown("### ✅ Features")
    st.write("✅ PDF/TXT official document upload")
    st.write("✅ English, Hindi, Telugu keyword support")
    st.write("✅ Citizen-service question answering")
    st.write("✅ Safety guardrail")
    st.write("✅ Source context display")

if uploaded_file:
    text = extract_text(uploaded_file)

    if not text.strip():
        st.error("No readable text found in this document.")
        st.stop()

    st.success("✅ Official document uploaded successfully!")

    question = st.text_input("🔎 Ask your citizen-service question")

    if question:
        if not guardrail_check(question):
            st.error("Request refused because it violates citizen-service safety rules.")
        else:
            answer, sources = search_answer(text, question)

            if language == "Hindi":
                final_answer = "आधिकारिक दस्तावेजों के आधार पर:\n\n" + answer
            elif language == "Telugu":
                final_answer = "అధికారిక పత్రాల ఆధారంగా:\n\n" + answer
            else:
                final_answer = "Based on official documents:\n\n" + answer

            st.markdown("### 🤖 Answer")
            st.markdown(
                f"""
                <div class='main-box'>
                    <p style='font-size:17px;line-height:1.7;'>{final_answer}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if sources:
                st.markdown("### 📌 Retrieved Sources")
                for i, (score, para) in enumerate(sources, start=1):
                    st.markdown(
                        f"""
                        <div class='source-box'>
                            <b>Source {i}</b> | Match Score: {score}<br><br>
                            {para}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        with st.expander("📄 View Document Preview"):
            st.text_area("Preview", text[:4000], height=300)

else:
    st.info("Upload an official PDF or TXT document to begin.")
