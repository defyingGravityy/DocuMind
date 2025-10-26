# 📄 DocuMind - Intelligent Document Assistant

An AI-powered document analysis system that enables smart Q&A on PDFs, CSVs, and text files using LangChain and Groq API.

# 🚀 Live Demo

**Try it here:** [DocuMind - Live App](https://ena7gudf2ueupuqe8dymzz.streamlit.app/)
## 🚀 Features

- **Document Q&A**: Upload PDF, TXT, DOCX and ask natural language questions
- **CSV SQL Queries**: Convert CSV to SQLite and query with natural language
- **Multi-language Support**: Automatic language detection
- **Real-time Chat**: Interactive conversation with source citations

## 🔧 Tech Stack

- **LangChain 0.2+** - RAG pipeline
- **Groq API** - Fast LLM inference
- **Chroma** - Vector database
- **Streamlit** - Web interface
- **HuggingFace** - Embeddings

## 📦 Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Create .env with your Groq API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

## 🚀 Usage
```bash
streamlit run app.py
```

Then open http://localhost:8501

# 🚀 Live Demo

**Try it here:** [DocuMind - Live App](https://ena7gudf2ueupuqe8dymzz.streamlit.app/)

## 📝 Project Structure
```
DocuMind/
├── main.py              # Entry point
├── app.py               # Document Q&A module
├── core.py              # RAG logic
├── requirements.txt     # Dependencies
├── .env                 # API keys (git ignored)
├── .gitignore          # Git ignore rules
└── README.md           # This file

```
