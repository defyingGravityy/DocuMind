"""
app.py - OOP Streamlit Application
Uses OOP models and services for Document Q&A and CSV/SQL features
"""

import streamlit as st
import os
import datetime
import sqlite3
import pandas as pd
from dotenv import load_dotenv

from models import (
    RegisteredUser, GuestUser, PDFDocument, CSVDocument, 
    WordDocument, TextDocument, Query, SQLQuery, Response, Session
)
from services import (
    DocumentManager, QueryManager, SessionManager, 
    AnalysisEngine, CSVToSQLConverter
)
from core import generate_sql_query, detect_language

load_dotenv()

# ============= PAGE CONFIG =============
st.set_page_config(
    page_title="DocuMind - OOP Version",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("DocuMind - Intelligent Document Assistant")
st.markdown("---")

# ============= INITIALIZE MANAGERS =============
if "doc_manager" not in st.session_state:
    st.session_state.doc_manager = DocumentManager()

if "query_manager" not in st.session_state:
    st.session_state.query_manager = QueryManager()

if "session_manager" not in st.session_state:
    st.session_state.session_manager = SessionManager()

if "analysis_engine" not in st.session_state:
    st.session_state.analysis_engine = AnalysisEngine()

if "current_user" not in st.session_state:
    st.session_state.current_user = RegisteredUser(
        "user_001", "user@example.com", "Demo User", "password123"
    )

if "current_session" not in st.session_state:
    st.session_state.current_session = st.session_state.session_manager.create_session(
        st.session_state.current_user.user_id
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_document" not in st.session_state:
    st.session_state.current_document = None

if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# ============= SIDEBAR =============
with st.sidebar:
    st.header("Configuration")
    
    # User Info
    st.subheader("User Info")
    st.write(f"**Name:** {st.session_state.current_user.name}")
    st.write(f"**Email:** {st.session_state.current_user.email}")
    st.write(f"**Subscription:** {st.session_state.current_user.subscription}")
    
    # Settings
    with st.expander("Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 200, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, step=50)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, step=0.1)
    
    # API Configuration
    st.subheader("API Setup")
    groq_api = os.getenv("GROQ_API_KEY", "")
    
    if not groq_api:
        st.error("GROQ_API_KEY not found!")
        st.info("Set it in .env file")
        st.stop()
    
    st.success("Groq API configured")

# ============= INITIALIZE ENGINE =============
if not st.session_state.analysis_engine.is_initialized:
    with st.spinner("Initializing Analysis Engine..."):
        st.session_state.analysis_engine.initialize(groq_api)
        st.success("Engine initialized!")

# ============= MAIN CONTENT =============
tab1, tab2 = st.tabs(["Document Q&A", "CSV Analysis"])

# ====== TAB 1: DOCUMENT Q&A ======
with tab1:
    st.subheader("Document Q&A")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, TXT, DOCX)",
        type=["pdf", "txt", "docx"]
    )
    
    if uploaded_file:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Upload document using DocumentManager
        if st.session_state.last_uploaded != uploaded_file.name:
            with st.spinner("Uploading document..."):
                doc = st.session_state.doc_manager.upload_document(
                    uploaded_file, 
                    st.session_state.current_user.user_id
                )
                
                # Add to user's documents
                st.session_state.current_user.documents.append(doc)
                st.session_state.current_user.documents_count += 1
                st.session_state.current_document = doc
                st.session_state.last_uploaded = uploaded_file.name
                st.session_state.chat_history = []
            
            st.success(f"Document uploaded: {doc.title}")
            st.info(f"Metadata: {doc.get_metadata()}")
        
        # Analyze document if not already done
        if st.session_state.current_document:
            file_path = os.path.join("temp_files", uploaded_file.name)
            
            if not st.session_state.analysis_engine.chain_dict:
                with st.spinner("Processing document..."):
                    chunks = st.session_state.analysis_engine.analyze_document(
                        st.session_state.current_document,
                        file_path,
                        chunk_size,
                        chunk_overlap
                    )
                    st.success(f"Document processed! ({len(chunks)} chunks)")
            
            # Question interface
            st.markdown("---")
            user_question = st.text_input(
                "Ask a question about the document:",
                placeholder="e.g., 'What is this document about?'"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                submit_btn = st.button("Ask Question", type="primary")
            with col2:
                clear_btn = st.button("Clear History")
            
            if clear_btn:
                st.session_state.chat_history = []
                st.session_state.analysis_engine.chain_dict = None
                st.rerun()
            
            if submit_btn and user_question.strip():
                with st.spinner("Thinking..."):
                    try:
                        # Create query using QueryManager
                        query = st.session_state.query_manager.create_query(
                            user_question,
                            st.session_state.current_user.user_id,
                            st.session_state.current_document.doc_id
                        )
                        
                        # Process query using AnalysisEngine
                        response = st.session_state.analysis_engine.process_query(query)
                        
                        # Add to session
                        st.session_state.current_session.add_query_response(query, response)
                        
                        # Store in chat history
                        st.session_state.chat_history.append({
                            "question": user_question,
                            "answer": response.get_response_text(),
                            "timestamp": datetime.datetime.now(),
                            "sources": response.get_sources(),
                            "confidence": response.get_confidence()
                        })
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            # Display results
            if st.session_state.chat_history:
                st.markdown("---")
                latest = st.session_state.chat_history[-1]
                
                st.markdown("### Assistant Response:")
                st.write(latest["answer"])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence", f"{latest['confidence']:.2%}")
                with col2:
                    st.metric("Sources", len(latest["sources"]))
                
                # Chat history
                with st.expander(f"Chat History ({len(st.session_state.chat_history)} messages)"):
                    for i, chat in enumerate(st.session_state.chat_history):
                        st.markdown(f"**Q {i+1}:** {chat['question']}")
                        st.markdown(f"**A {i+1}:** {chat['answer']}")
                        st.caption(f"{chat['timestamp'].strftime('%H:%M:%S')} | 🎯 {chat['confidence']:.2%}")
                        st.divider()
                
                # Sources
                if latest["sources"]:
                    with st.expander("Source Chunks"):
                        for i, source in enumerate(latest["sources"][:3]):
                            st.markdown(f"**Source {i+1}:**")
                            st.text(source[:500] + "..." if len(source) > 500 else source)

# ====== TAB 2: CSV ANALYSIS ======
with tab2:
    st.subheader("CSV Data Analysis")
    
    csv_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )
    
    if csv_file:
        file_path = os.path.join("temp_files", csv_file.name)
        
        if st.session_state.last_uploaded != csv_file.name:
            with st.spinner("Processing CSV..."):
                # Upload CSV as document
                csv_doc = st.session_state.doc_manager.upload_document(
                    csv_file,
                    st.session_state.current_user.user_id
                )
                st.session_state.current_document = csv_doc
                st.session_state.last_uploaded = csv_file.name
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(csv_file.getvalue())
            
            st.success(f"CSV uploaded: {csv_doc.title}")
        
        # CSV Preview
        df = pd.read_csv(file_path)
        with st.expander("Data Preview", expanded=False):
            st.dataframe(df.head(10))
            st.caption(f"Total rows: {len(df)} | Columns: {len(df.columns)}")
        
        # Convert to SQL
        if st.session_state.last_uploaded == csv_file.name:
            with st.spinner("Converting to SQLite..."):
                converter = CSVToSQLConverter(file_path)
                converter.convert_to_sql()
                st.success("Database ready!")
            
            # Query interface
            user_query = st.text_input(
                "Ask a question about your data:",
                placeholder="e.g., 'What is the total revenue?'"
            )
            
            if st.button("Execute Query", type="primary"):
                if user_query.strip():
                    with st.spinner("Generating SQL..."):
                        try:
                            # Create SQL query using QueryManager
                            sql_query_obj = st.session_state.query_manager.create_sql_query(
                                user_query,
                                st.session_state.current_user.user_id,
                                st.session_state.current_document.doc_id
                            )
                            
                            # Generate SQL using LLM
                            sql_text = st.session_state.query_manager.get_sql_query(
                                user_query,
                                "data",
                                st.session_state.analysis_engine.llm
                            )
                            
                            st.markdown("**Generated SQL:**")
                            st.code(sql_text, language="sql")
                            
                            # Execute SQL
                            conn = sqlite3.connect("user_data.db")
                            cursor = conn.cursor()
                            cursor.execute(sql_text)
                            result = cursor.fetchall()
                            columns = [desc[0] for desc in cursor.description]
                            conn.close()
                            
                            if result:
                                df_result = pd.DataFrame(result, columns=columns)
                                st.markdown("**Query Results:**")
                                st.dataframe(df_result, use_container_width=True)
                                
                                # Store response
                                response = Response(
                                    f"response_{datetime.datetime.now().timestamp()}",
                                    df_result.to_string(),
                                    sql_query_obj.query_id
                                )
                                
                                # Explain results
                                if st.button("Explain Results", type="secondary"):
                                    with st.spinner("Generating explanation..."):
                                        try:
                                            markdown_table = df_result.to_markdown(index=False)
                                            prompt = f"""You are a data analyst. 
Summarize this data with key insights:

{markdown_table}

Provide a brief explanation."""
                                            
                                            explanation = st.session_state.analysis_engine.llm.invoke(prompt)
                                            explanation_text = (
                                                explanation.content 
                                                if hasattr(explanation, "content") 
                                                else str(explanation)
                                            )
                                            st.markdown("**Analysis:**")
                                            st.write(explanation_text)
                                        except Exception as e:
                                            st.error(f"Error: {str(e)}")
                            else:
                                st.info("No results found.")
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please enter a question.")

# ============= SESSION INFO =============
with st.sidebar:
    st.markdown("---")
    st.subheader("Session Info")
    
    session = st.session_state.current_session
    st.metric("Session Duration", f"{session.get_duration()}s")
    st.metric("Messages", len(session.get_history()))
    st.metric("Documents", len(st.session_state.current_user.documents))

if __name__ == "__main__":
    pass