import streamlit as st
import os
import datetime
import sqlite3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from core import (
    run_rag_chain_with_sources,
    load_documents_from_files,
    detect_language,
    get_llm,
    get_embedding_model,
    get_or_create_vectorstore,
    build_conversational_chain,
    convert_csv_to_sqlite,
    generate_sql_query,
    DEFAULT_EMBED_MODEL,
    MULTILINGUAL_EMBED_MODEL,
    DEFAULT_LLM_MODELS,
)

load_dotenv()

def run_app():
    st.set_page_config(
        page_title="DocuMind: Understand Your Documents, Effortlessly",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("DocuMind: Understand Your Documents, Effortlessly")
    st.markdown("---")

    # === Sidebar Configuration ===
    with st.sidebar:
        st.header("Configuration")
        
        with st.expander("Advanced Settings", expanded=False):
            chunk_size = st.slider("Chunk Size", 200, 2000, 1000, step=100)
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, step=50)
            temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.3, step=0.1)

        embed_choice = st.selectbox(
            "Embedding Model",
            [DEFAULT_EMBED_MODEL, MULTILINGUAL_EMBED_MODEL]
        )
        llm_choice = st.selectbox("LLM Model", DEFAULT_LLM_MODELS)

    # === Session State Initialization ===
    if "store" not in st.session_state:
        st.session_state.store = {}
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "chain_dict" not in st.session_state:
        st.session_state.chain_dict = None
    if "document_ready" not in st.session_state:
        st.session_state.document_ready = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_uploaded" not in st.session_state:
        st.session_state.last_uploaded = None

    # === API Key Setup ===
    is_groq = llm_choice.startswith("Groq")

    if is_groq:
        groq_api = os.getenv("GROQ_API_KEY", "")
        if not groq_api:
            st.error("GROQ_API_KEY not found in environment variables!")
            st.info("**How to set up:**")
            st.code("export GROQ_API_KEY='your_key_here'  # Linux/Mac")
            st.code("set GROQ_API_KEY=your_key_here      # Windows")
            st.stop()
        st.session_state.api_key = groq_api
        
        st.success("Groq API configured")
        st.info("""
        ### Welcome to DocuMind - BDIA Platform!
        
        **Intelligent document understanding at your fingertips**
        - Supports: PDF, CSV, TXT, DOCX
        - Multi-language support
        - Smart Q&A with source tracking
        - SQL-based data queries
        """)
    else:
        st.session_state.api_key = "local"
        st.info(f"Using local LLM model: `{llm_choice}`")

    # === Main Application ===
    if st.session_state.api_key:
        try:
            llm = get_llm(st.session_state.api_key, llm_choice, temperature)
        except Exception as e:
            st.error(f"Failed to initialize LLM: {str(e)}")
            st.stop()

        # === File Upload ===
        st.subheader("Upload Your Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "csv", "txt", "docx"]
        )

        if uploaded_file:
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            
            # Create temp directory
            os.makedirs("temp_files", exist_ok=True)
            file_path = os.path.join("temp_files", uploaded_file.name)

            # Save uploaded file
            if st.session_state.last_uploaded != uploaded_file.name:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                st.session_state.document_ready = False
                st.session_state.last_uploaded = uploaded_file.name
                st.session_state.chat_history = []

            st.success(f"File uploaded: `{uploaded_file.name}`")

            # === CSV Flow ===
            if file_ext == ".csv":
                st.subheader("CSV Data Analysis")
                
                with st.spinner("Converting CSV to SQLite..."):
                    try:
                        db_path = "user_data.db"
                        db_uri, table_name = convert_csv_to_sqlite(file_path, db_path=db_path)
                        st.success("Database ready!")
                        
                        # Show data preview
                        df = pd.read_csv(file_path)
                        with st.expander("Data Preview", expanded=False):
                            st.dataframe(df.head(10))
                            st.caption(f"Total rows: {len(df)} | Columns: {len(df.columns)}")
                        
                        # Query interface
                        user_query = st.text_input(
                            "Ask a question about your data:",
                            placeholder="e.g., 'What is the total revenue?' or 'Show top 5 customers'"
                        )

                        if st.button("Execute Query", type="primary"):
                            if user_query.strip():
                                with st.spinner("Generating SQL and fetching results..."):
                                    try:
                                        sql_query = generate_sql_query(
                                            llm, user_query, table_name, db_path
                                        )
                                        
                                        st.markdown("**Generated SQL:**")
                                        st.code(sql_query, language="sql")

                                        # Execute query
                                        conn = sqlite3.connect(db_path)
                                        cursor = conn.cursor()
                                        cursor.execute(sql_query)
                                        result = cursor.fetchall()
                                        columns = [desc[0] for desc in cursor.description]
                                        conn.close()

                                        if result:
                                            df_result = pd.DataFrame(result, columns=columns)
                                            st.markdown("**Query Results:**")
                                            st.dataframe(df_result, use_container_width=True)

                                            # Natural language explanation
                                            if st.button("Explain Results", type="secondary"):
                                                with st.spinner("Generating explanation..."):
                                                    try:
                                                        markdown_table = df_result.to_markdown(index=False)
                                                        prompt = f"""You are a data analyst. 
Summarize this data with key insights and trends:

{markdown_table}

Provide a brief, clear explanation."""
                                                        explanation = llm.invoke(prompt)
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
                                            st.info("No results found for this query.")
                                    except Exception as e:
                                        st.error(f"SQL Error: {str(e)}")
                            else:
                                st.warning("Please enter a question.")
                    except Exception as e:
                        st.error(f"Error processing CSV: {str(e)}")

            # === PDF/TXT/DOCX Flow (RAG) ===
            elif file_ext in [".pdf", ".txt", ".docx"]:
                st.subheader("Document Q&A")
                
                if not st.session_state.document_ready:
                    with st.spinner("Processing document (loading + embedding)..."):
                        try:
                            # Load documents
                            documents = load_documents_from_files([file_path])
                            if not documents:
                                st.error("Could not read document.")
                                st.stop()

                            # Detect language
                            sample_text = documents[0].page_content[:1000]
                            detected_lang = detect_language(sample_text)
                            st.info(f"Detected Language: `{detected_lang.upper()}`")

                            # Choose embedding model
                            embedding_model_name = embed_choice
                            if detected_lang != "en":
                                embedding_model_name = MULTILINGUAL_EMBED_MODEL
                                st.warning("Using multilingual embedding model.")

                            # Create vectorstore
                            embeddings = get_embedding_model(embedding_model_name)
                            vectorstore = get_or_create_vectorstore(
                                filepath=file_path,
                                documents=documents,
                                embeddings=embeddings,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                persist_directory="chroma_store",
                            )

                            # Build chain
                            retriever = vectorstore.as_retriever(
                                search_kwargs={"k": 3}
                            )
                            chain_dict = build_conversational_chain(
                                llm, retriever, session_id="default"
                            )

                            st.session_state.chain_dict = chain_dict
                            st.session_state.document_ready = True
                            st.success("Document processed! Ready for questions.")
                        except Exception as e:
                            st.error(f"Error processing document: {str(e)}")
                            st.stop()
                else:
                    st.success("Document already processed.")

                # === Question Interface ===
                if st.session_state.chain_dict:
                    user_question = st.text_input(
                        "Ask a question about the document:",
                        placeholder="e.g., 'What are the main points?' or 'Summarize chapter 2'"
                    )

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        submit_btn = st.button("Ask Question", type="primary")
                    with col2:
                        clear_btn = st.button("Clear History")

                    if clear_btn:
                        st.session_state.chat_history = []
                        if st.session_state.chain_dict:
                            st.session_state.chain_dict["chat_history"].clear()
                        st.rerun()

                    if submit_btn and user_question.strip():
                        with st.spinner("Thinking..."):
                            try:
                                response = run_rag_chain_with_sources(
                                    st.session_state.chain_dict,
                                    user_question
                                )
                                
                                if response["success"]:
                                    answer = response["answer"]
                                    timestamp = datetime.datetime.now()
                                    st.session_state.chat_history.append(
                                        (user_question, answer, timestamp)
                                    )
                                else:
                                    st.error(answer)
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

                    # === Display Results ===
                    if st.session_state.chat_history:
                        st.markdown("---")
                        latest_q, latest_a, _ = st.session_state.chat_history[-1]
                        
                        col1, col2 = st.columns([1, 5])
                        with col1:
                            st.markdown("**Assistant:**")
                        with col2:
                            st.markdown(latest_a)

                        # Chat History
                        with st.expander(f"Chat History ({len(st.session_state.chat_history)} messages)"):
                            for i, (q, a, t) in enumerate(st.session_state.chat_history):
                                time_str = t.strftime("%H:%M:%S")
                                st.markdown(f"**[{time_str}] Q:** {q}")
                                st.markdown(f"**A:** {a}")
                                st.divider()

    else:
        st.info("Please configure your API key to continue.")


if __name__ == "__main__":
    run_app()