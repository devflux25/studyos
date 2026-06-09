# 🎓 StudyOS — AI Study Companion

Tired of making flashcards manually? StudyOS lets you upload your notes 
and instantly get quizzed, track your weak topics, and get a personalized 
study plan — all powered by AI.

> Built this because I was struggling to study from 50-page PDFs before 
> exams and couldn't find a tool that actually understood my own notes.

## 🔗 Live Demo
[Click here to try StudyOS](https://studyos-kairav.streamlit.app/)

## ✨ Features
- 📄 Upload any PDF notes or PYQs
- 💬 Ask questions and get answers from your own notes
- 🧠 Auto-generate MCQs from your study material
- 📊 Track your quiz scores and weak topics over time
- 📅 Get a personalized 3-day study plan based on where you're struggling

## 🛠️ Built With
Python · Streamlit · LangChain · FAISS · Google Gemini API · SQLite

## 🚀 Run Locally
git clone https://github.com/devflux25/studyos
cd studyos
pip install -r requirements.txt

Create a .env file:
GEMINI_API_KEY=your-key-here

streamlit run app.py

## 👨‍💻 About
Built by Kairav Joshi — 1st year CSE student at Manipal University Jaipur.
