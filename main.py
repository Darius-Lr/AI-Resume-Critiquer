import streamlit as st
import PyPDF2
import io
from transformers import pipeline

st.set_page_config(page_title="AI Resume Critiquer", page_icon="📃", layout="centered")

st.title("AI Resume Critiquer")
st.markdown("Upload your resume and get AI-powered feedback tailored to your needs!")

uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
job_role = st.text_input("Enter the job role you're targeting (optional)")
analyze = st.button("Analyze Resume")


def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("utf-8")


def chunk_text(text, chunk_size=2000):
    """Split long text into chunks to avoid token limits."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="bigscience/bloom-560m",
        use_auth_token=True,  
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7
    )


if analyze and uploaded_file:
    try:
        file_content = extract_text_from_file(uploaded_file)
        if not file_content.strip():
            st.error("File does not have any content...")
            st.stop()

        generator = load_model()

    
        job_text = f"Targeted job role: {job_role}\n" if job_role else ""

        feedback = ""
        for chunk in chunk_text(file_content, chunk_size=2000):
            prompt = f"""
You are an AI career advisor. Analyze this resume section and provide feedback in numbered bullet points:
1. Content clarity
2. Skills presentation
3. Experience descriptions
4. Specific suggestions for improvement

{job_text}

Resume section:
{chunk}
"""
            output = generator(prompt)
            feedback += output[0]["generated_text"] + "\n\n"

        st.markdown("### Analysis Results")
        st.markdown(feedback)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
