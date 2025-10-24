import os
import hashlib
import pandas as pd
import sqlite3
from typing import List, Dict, Any
from langdetect import detect

# === LangChain v0.2+ imports (Updated) ===
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredCSVLoader,
)

# Utilities
from langchain_community.utilities import SQLDatabase

# === Defaults ===
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MULTILINGUAL_EMBED_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v1"
DEFAULT_LLM_MODELS = [
    "Groq - llama-3.1-8b-instant",
    "Groq - llama-3.1-70b-versatile",
    "Groq - mixtral-8x7b-32768",
    "Local - llama2",
    "Local - mistral",
]

# === Simple In-Memory Chat History ===
class SimpleChatHistory(BaseChatMessageHistory):
    """Simple in-memory chat history implementation"""
    def __init__(self):
        self.messages: List[BaseMessage] = []

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)

    def clear(self) -> None:
        self.messages.clear()

# === File Hashing ===
def calculate_file_hash(filepath: str) -> str:
    """Calculate MD5 hash of file for unique collection naming"""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# === Language Detection ===
def detect_language(text: str) -> str:
    """Detect language of text"""
    try:
        return detect(text)
    except Exception:
        return "unknown"

# === Document Loaders ===
def load_documents_from_files(file_paths: List[str]):
    """Load documents from various file formats"""
    docs = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".csv":
                loader = UnstructuredCSVLoader(file_path)
            elif ext in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif ext == ".txt":
                loader = TextLoader(file_path)
            else:
                print(f"Unsupported file type: {file_path}")
                continue
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            continue
    return docs

# === Embeddings ===
def get_embedding_model(model_name: str = DEFAULT_EMBED_MODEL):
    """Initialize HuggingFace embeddings"""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"trust_remote_code": True}
    )

# === Document Chunking ===
def split_documents(documents, chunk_size: int = 800, chunk_overlap: int = 100):
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)

# === Vectorstore ===
def get_or_create_vectorstore(
    filepath: str,
    documents,
    embeddings,
    chunk_size: int,
    chunk_overlap: int,
    persist_directory: str = "chroma_store",
):
    """Create or retrieve vectorstore from Chroma"""
    file_hash = calculate_file_hash(filepath)
    collection_name = f"collection_{file_hash}"
    vectordb_path = os.path.join(persist_directory, collection_name)

    try:
        # Try to load existing collection
        if os.path.exists(vectordb_path):
            return Chroma(
                persist_directory=persist_directory,
                collection_name=collection_name,
                embedding_function=embeddings,
            )
    except Exception as e:
        print(f"Could not load existing vectorstore: {str(e)}")

    # Create new collection
    os.makedirs(persist_directory, exist_ok=True)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    
    if not chunks:
        raise ValueError("No document chunks created")
    
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

# === Prompts ===
def get_contextualize_prompt() -> ChatPromptTemplate:
    """Prompt for contextualizing questions based on chat history"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", 
             "Given a chat history and the latest user question, formulate a standalone question "
             "which can be understood without the chat history. Do not answer the question, just "
             "reformulate it if needed; otherwise, return it as is."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

def get_qa_prompt() -> ChatPromptTemplate:
    """Prompt for question-answering with context"""
    return ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a helpful assistant for question-answering tasks. "
             "Use the following pieces of retrieved context to answer the question. "
             "If you don't know the answer, say that you don't know. "
             "Keep answers concise and relevant.\n\nContext: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

# === RAG Chain (Updated for v0.2+) ===
def build_conversational_chain(llm, retriever, session_id: str = "default"):
    """
    Build a RAG chain using LangChain v0.2+
    Returns a callable chain that handles Q&A with context
    """
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    
    # Initialize chat history
    chat_history = SimpleChatHistory()
    
    # Get prompts
    qa_prompt = get_qa_prompt()
    
    # Build RAG chain
    def format_docs(docs):
        """Format retrieved documents into a string"""
        if not docs:
            return "No relevant documents found."
        if isinstance(docs, list):
            return "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs])
        return str(docs)
    
    # Simple chain that works reliably
    def chain_func(input_query: str) -> str:
        """Execute the chain with proper error handling"""
        try:
            # Retrieve documents
            retrieved_docs = retriever.invoke(input_query)
            formatted_context = format_docs(retrieved_docs)
            
            # Build the prompt
            prompt_value = qa_prompt.invoke({
                "context": formatted_context,
                "chat_history": chat_history.messages,
                "input": input_query
            })
            
            # Get response from LLM
            response = llm.invoke(prompt_value)
            
            # Extract text from response
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            return answer
        except Exception as e:
            raise Exception(f"Chain execution failed: {str(e)}")
    
    return {
        "chain": chain_func,
        "retriever": retriever,
        "chat_history": chat_history,
    }

# === CSV-to-SQL ===
def convert_csv_to_sqlite(
    csv_path: str,
    db_path: str = "user_data.db",
    table_name: str = "data"
) -> tuple:
    """Convert CSV to SQLite database"""
    try:
        df = pd.read_csv(csv_path)
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        return f"sqlite:///{os.path.abspath(db_path)}", table_name
    except Exception as e:
        raise Exception(f"Error converting CSV to SQLite: {str(e)}")

def generate_sql_query(
    llm,
    user_question: str,
    table_name: str,
    db_path: str = "user_data.db"
) -> str:
    """Generate SQL query from natural language question"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()

        if not columns:
            raise ValueError(f"Table '{table_name}' not found or has no columns")

        column_info = ", ".join([f'"{col}"' for col in columns])
        
        prompt = f"""You are an expert SQL assistant.
The table '{table_name}' has the following columns: {column_info}

Write a valid SQLite query to answer this question:
"{user_question}"

Rules:
- Always wrap column names in double quotes
- Use valid SQLite syntax
- Return ONLY the SQL query, no explanation

SQL Query:"""
        
        response = llm.invoke(prompt)
        # Handle both string and object responses
        sql_query = response.content if hasattr(response, "content") else str(response)
        return sql_query.strip()
    except Exception as e:
        raise Exception(f"Error generating SQL query: {str(e)}")

# === LLM initialization ===
def get_llm(api_key: str, model_name: str, temperature: float = 0.3):
    """Initialize LLM based on model name"""
    try:
        if model_name.startswith("Groq"):
            actual_model = model_name.split(" - ")[-1]
            return ChatGroq(
                groq_api_key=api_key,
                model_name=actual_model,
                temperature=temperature,
            )
        elif model_name.startswith("Local"):
            try:
                from langchain_community.llms import Ollama
                actual_model = model_name.split(" - ")[-1]
                return Ollama(model=actual_model, temperature=temperature)
            except ImportError:
                raise ImportError(
                    "Ollama not available. Install: pip install langchain-community"
                )
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
    except Exception as e:
        raise Exception(f"Error initializing LLM: {str(e)}")

# === Run Chain ===
def run_rag_chain_with_sources(chain_dict: Dict, input_query: str) -> Dict[str, Any]:
    """Run RAG chain and return answer with source documents"""
    answer = "No answer generated"
    docs = []
    
    try:
        chain_func = chain_dict["chain"]
        retriever = chain_dict["retriever"]
        chat_history = chain_dict["chat_history"]
        
        # Get relevant documents
        try:
            docs = retriever.invoke(input_query)
        except Exception as e:
            docs = []
            print(f"Retriever error: {str(e)}")
        
        # Run the chain
        try:
            answer = chain_func(input_query)
        except Exception as e:
            print(f"Chain error: {str(e)}")
            raise
        
        # Update chat history
        from langchain_core.messages import HumanMessage, AIMessage
        try:
            chat_history.add_message(HumanMessage(content=input_query))
            chat_history.add_message(AIMessage(content=answer))
        except Exception as e:
            print(f"Chat history error: {str(e)}")
        
        return {
            "answer": answer,
            "context": docs,
            "success": True,
        }
    except Exception as e:
        print(f"Final error: {str(e)}")
        return {
            "answer": f"Error: {str(e)}",
            "context": docs,
            "success": False,
        }