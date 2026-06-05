import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
from datetime import date
import sqlite3 
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

#st.set_page_config(page_title="StudyOS", page_icon="🎓")
#st.title("🎓 StudyOS")
#st.write("Your AI-powered study companion")

def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text 

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def answer_question(question):
    embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    docs = vector_store.similarity_search(question)

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer this question based on the context below:\n\nContext: {context}\n\nQuestion: {question}"

    response = model.invoke(prompt)
    if hasattr(response, 'content'):
        return response.content
    else:
        return str(response)
    

def mcq_generator(text):
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=1.0
    )
    prompt = f"""You are a smart study assistant. Generate 5 MCQs to test conceptual understanding.

    STRICT RULES:
    - Only ask about concepts, theories, laws, formulas and definitions
    - NEVER ask about author names, dates, course codes, batch years, exam schedules
    - Questions must test if the student actually understands the topic
    - Each question must be different from the others

    Return ONLY a JSON array, no extra text:
    [
    {{
        "question": "...",
        "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
        "answer": "A"
    }}
    ]

    Text: {text[:3000]}"""
    response = model.invoke(prompt)
    import json
    import re

    raw = response.content

    match = re.search(r'\[.*\]',raw,re.DOTALL)
    if match:
        return json.loads(match.group())
    return []

def weak_topic_tracker(score,total,subject):
    conn = sqlite3.connect('my_database.db')

    cursor = conn.cursor()   
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS  quiz_results(
        subject TEXT,
        score INTEGER,
        total INTEGER,
        date TEXT
    )
    """)

    conn.commit()
    today = str(date.today())

    # What to do with subject how to input them 
    cursor.execute("""
        INSERT INTO quiz_results (subject, score, total, date)
        VALUES (?, ?, ?, ?)
    """, ( subject ,score, total, today))

    conn.commit()
    conn.close()
 

def show_analytics():
    conn = sqlite3.connect('my_database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT subject, score, total, date FROM quiz_results")
    results = cursor.fetchall()
    conn.close()
    return results


def main():
    st.set_page_config(page_title="StudyOS", page_icon="🎓")
    st.title("🎓 StudyOS")
    st.write("Your AI-powered study companion")

    # sidebar - upload PDF
    with st.sidebar:
        st.subheader("Upload Your Notes")
        pdf_file = st.file_uploader("Upload a PDF", type="pdf")
        
        if st.button("Process PDF"):
            if pdf_file is None:
                st.warning("Please upload a PDF first!")
            else:
                with st.spinner("Reading your notes..."):
                    raw_text = get_pdf_text(pdf_file)
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                    st.success("PDF processed! You can now ask questions.")


    # main area - ask questions
    question = st.text_input("Ask a question from your notes:")
    
    if st.button("Get Answer"):
        if question.strip() == "":
            st.warning("Please type a question!")
        else:
            with st.spinner("Finding answer in your notes..."):
                try:
                    answer = answer_question(question)
                    st.write(answer)
                except Exception:
                    st.error("Please upload and process a PDF first!")

    st.divider()
    st.subheader("🧠 Quiz Mode")
    subject = st.text_input("Enter subject name (e.g. Physics)")
    if st.button("Generate MCQs from my notes"):
        if not os.path.exists("faiss_index"):
            st.warning("Please upload and process a PDF first!")
        else:
            with st.spinner("Generating questions..."):
                try:
                    # get text from uploaded pdf
                    raw_text = get_pdf_text(pdf_file)
                    mcqs = mcq_generator(raw_text)
                    st.session_state.mcqs = mcqs
                    st.session_state.score = 0
                    st.session_state.answered = [None] * len(mcqs)
                except Exception as e:
                    st.error(f"Error: {e}")

    # display MCQs
    if "mcqs" in st.session_state and st.session_state.mcqs:
        st.write(f"**Total questions: {len(st.session_state.mcqs)}**")
        
        for i, mcq in enumerate(st.session_state.mcqs):
            st.write(f"**Q{i+1}. {mcq['question']}**")
            answer = st.radio(
                "Choose your answer:",
                mcq['options'],
                key=f"q{i}"
            )
            st.session_state.answered[i] = answer
        
        if st.button("Submit Quiz"):
            score = 0
            for i, mcq in enumerate(st.session_state.mcqs):
                selected = st.session_state.answered[i]
                correct = mcq['answer']
                if selected and selected.startswith(correct):
                    score += 1
                    st.success(f"Q{i+1} ✅ Correct!")
                else:
                    st.error(f"Q{i+1} ❌ Wrong! Correct answer: {correct}")
            
            st.write(f"## Your Score: {score}/{len(st.session_state.mcqs)}")

            total = len(st.session_state.mcqs)

            weak_topic_tracker(score,total,subject)

    st.divider()
    st.subheader("📊 Your Progress")

    if st.button("Show My Progress"):
        results = show_analytics()
        if len(results) == 0:
            st.info("No quiz attempts yet. Take a quiz first!")
        else:
            import pandas as pd
            df = pd.DataFrame(results, columns=["Subject", "Score", "Total", "Date"])
            df["Percentage"] = (df["Score"] / df["Total"] * 100).round(1)
            st.dataframe(df)
            st.bar_chart(df.set_index("Date")["Percentage"])


if __name__ == "__main__":
    main()




