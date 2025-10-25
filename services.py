"""
services.py - Business Logic & Manager Classes
Contains DocumentManager, QueryManager, SessionManager, and AnalysisEngine
"""

import os
import sqlite3
from typing import List, Dict, Optional
from datetime import datetime
from models import (
    RegisteredUser, Document, PDFDocument, CSVDocument, 
    WordDocument, TextDocument, Query, SQLQuery, Response, Session, ChatMessage
)

# Import from core.py (existing functionality)
from core import (
    load_documents_from_files,
    get_llm,
    get_embedding_model,
    get_or_create_vectorstore,
    build_conversational_chain,
    run_rag_chain_with_sources,
    convert_csv_to_sqlite,
    generate_sql_query,
)


class DocumentManager:
    """Manages document operations"""
    
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.upload_path = "temp_files"
        os.makedirs(self.upload_path, exist_ok=True)
    
    def upload_document(self, file, user_id: str) -> Document:
        """Upload a document"""
        file_ext = os.path.splitext(file.name)[1].lower()
        doc_id = f"doc_{len(self.documents)}_{datetime.now().timestamp()}"
        
        # Save file
        file_path = os.path.join(self.upload_path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        
        # Create appropriate document object
        if file_ext == ".pdf":
            doc = PDFDocument(doc_id, file.name, user_id)
        elif file_ext == ".csv":
            doc = CSVDocument(doc_id, file.name, user_id)
        elif file_ext == ".docx":
            doc = WordDocument(doc_id, file.name, user_id)
        else:
            doc = TextDocument(doc_id, file.name, user_id)
        
        doc.file_size = os.path.getsize(file_path)
        self.documents[doc_id] = doc
        
        return doc
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            return True
        return False
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID"""
        return self.documents.get(doc_id)
    
    def get_user_documents(self, user_id: str) -> List[Document]:
        """Get all documents of a user"""
        return [doc for doc in self.documents.values() if doc.uploaded_by == user_id]
    
    def load_document(self, doc_id: str) -> Optional[Document]:
        """Load a document"""
        return self.get_document(doc_id)


class QueryManager:
    """Manages query operations"""
    
    def __init__(self):
        self.queries: Dict[str, Query] = {}
        self.responses: Dict[str, Response] = {}
        self.query_count = 0
    
    def create_query(self, query_text: str, user_id: str, doc_id: Optional[str] = None) -> Query:
        """Create a new query"""
        query_id = f"query_{self.query_count}_{datetime.now().timestamp()}"
        self.query_count += 1
        
        query = Query(query_id, query_text, user_id, doc_id)
        self.queries[query_id] = query
        
        return query
    
    def create_sql_query(self, query_text: str, user_id: str, doc_id: str) -> SQLQuery:
        """Create a SQL query"""
        query_id = f"sqlquery_{self.query_count}_{datetime.now().timestamp()}"
        self.query_count += 1
        
        sql_query = SQLQuery(query_id, query_text, user_id, doc_id)
        self.queries[query_id] = sql_query
        
        return sql_query
    
    def execute_query(self, query: Query, llm, retriever) -> Response:
        """Execute a query using RAG"""
        query.set_status("processing")
        
        # Use existing RAG chain
        chain_dict = {
            "chain": lambda q: self._rag_execute(q, llm, retriever),
            "retriever": retriever,
        }
        
        result = run_rag_chain_with_sources(chain_dict, query.query_text)
        
        response_id = f"response_{datetime.now().timestamp()}"
        response = Response(response_id, result["answer"], query.query_id)
        response.sources = [doc.page_content if hasattr(doc, 'page_content') else str(doc) 
                           for doc in result.get("context", [])]
        
        self.responses[response_id] = response
        query.set_status("completed")
        
        return response
    
    def _rag_execute(self, query_text: str, llm, retriever) -> str:
        """Execute RAG internally"""
        docs = retriever.invoke(query_text)
        context = "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs])
        
        prompt = f"""You are a helpful assistant. Answer the question based on the context below.

Context: {context}

Question: {query_text}

Answer:"""
        
        response = llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def get_sql_query(self, nlp_text: str, table_name: str, llm) -> str:
        """Generate SQL query from natural language"""
        sql_query = generate_sql_query(llm, nlp_text, table_name)
        return sql_query
    
    def get_query_history(self, user_id: str) -> List[Query]:
        """Get query history for a user"""
        return [q for q in self.queries.values() if q.user_id == user_id]
    
    def delete_query(self, query_id: str) -> bool:
        """Delete a query"""
        if query_id in self.queries:
            del self.queries[query_id]
            return True
        return False


class SessionManager:
    """Manages user sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.active_sessions = 0
    
    def create_session(self, user_id: str) -> Session:
        """Create a new session"""
        session_id = f"session_{user_id}_{datetime.now().timestamp()}"
        session = Session(session_id, user_id)
        self.sessions[session_id] = session
        self.active_sessions += 1
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session"""
        return self.sessions.get(session_id)
    
    def end_session(self, session_id: str) -> None:
        """End a session"""
        if session_id in self.sessions:
            self.sessions[session_id].end_session()
            self.active_sessions -= 1
    
    def get_all_user_sessions(self, user_id: str) -> List[Session]:
        """Get all sessions of a user"""
        return [s for s in self.sessions.values() if s.user_id == user_id]
    
    def save_session(self, session: Session) -> bool:
        """Save session (persist if needed)"""
        self.sessions[session.session_id] = session
        return True
    
    def add_message(self, session_id: str, sender: str, content: str) -> ChatMessage:
        """Add message to session"""
        session = self.get_session(session_id)
        if session:
            message_id = f"msg_{datetime.now().timestamp()}"
            message = ChatMessage(message_id, session_id, sender, content)
            session.add_message(message)
            return message
        return None


class AnalysisEngine:
    """Main analysis engine for processing documents and queries"""
    
    def __init__(self, llm_model_name: str = "Groq - llama-3.1-8b-instant", 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.engine_id = f"engine_{datetime.now().timestamp()}"
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.is_initialized = False
        
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.chain_dict = None
    
    def initialize(self, api_key: str) -> bool:
        """Initialize the analysis engine"""
        try:
            # Initialize LLM
            self.llm = get_llm(api_key, self.llm_model_name, temperature=0.3)
            
            # Initialize embeddings
            self.embeddings = get_embedding_model(self.embedding_model_name)
            
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Error initializing engine: {str(e)}")
            return False
    
    def analyze_document(self, doc: Document, file_path: str, chunk_size: int = 1000, 
                        chunk_overlap: int = 200) -> List[str]:
        """Analyze and chunk a document"""
        try:
            # Load documents
            documents = load_documents_from_files([file_path])
            
            if not documents:
                return []
            
            # Create vectorstore
            self.vectorstore = get_or_create_vectorstore(
                filepath=file_path,
                documents=documents,
                embeddings=self.embeddings,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                persist_directory="chroma_store"
            )
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Build chain
            self.chain_dict = build_conversational_chain(self.llm, self.retriever)
            
            # Return chunks
            chunks = [d.page_content for d in documents]
            return chunks
        except Exception as e:
            print(f"Error analyzing document: {str(e)}")
            return []
    
    def process_query(self, query: Query) -> Response:
        """Process a query using RAG"""
        try:
            if not self.chain_dict:
                raise Exception("Analysis engine not initialized with a document")
            
            result = run_rag_chain_with_sources(self.chain_dict, query.query_text)
            
            response_id = f"response_{datetime.now().timestamp()}"
            response = Response(response_id, result["answer"], query.query_id)
            response.sources = [doc.page_content if hasattr(doc, 'page_content') else str(doc) 
                               for doc in result.get("context", [])]
            
            query.set_status("completed")
            return response
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            response_id = f"response_{datetime.now().timestamp()}"
            return Response(response_id, f"Error: {str(e)}", query.query_id)
    
    def generate_embedding(self, text: str):
        """Generate embedding for text"""
        if not self.embeddings:
            return None
        return self.embeddings.embed_query(text)
    
    def shutdown(self) -> None:
        """Shutdown the engine"""
        self.is_initialized = False
        print(f"Engine {self.engine_id} shutdown")


class CSVToSQLConverter:
    """Convert CSV to SQL"""
    
    def __init__(self, csv_file, table_name: str = "data", db_path: str = "user_data.db"):
        self.csv_file = csv_file
        self.table_name = table_name
        self.db_path = db_path
    
    def parse_csv(self) -> List[Dict]:
        """Parse CSV file"""
        import pandas as pd
        df = pd.read_csv(self.csv_file)
        return df.to_dict(orient='records')
    
    def infer_schema(self) -> Dict[str, str]:
        """Infer schema from CSV"""
        import pandas as pd
        df = pd.read_csv(self.csv_file)
        schema = {}
        for col, dtype in df.dtypes.items():
            if dtype == 'object':
                schema[col] = 'TEXT'
            elif dtype == 'int64':
                schema[col] = 'INTEGER'
            elif dtype == 'float64':
                schema[col] = 'REAL'
            else:
                schema[col] = 'TEXT'
        return schema
    
    def generate_create_table_sql(self) -> str:
        """Generate CREATE TABLE SQL"""
        schema = self.infer_schema()
        columns = ", ".join([f'"{col}" {dtype}' for col, dtype in schema.items()])
        return f'CREATE TABLE "{self.table_name}" ({columns})'
    
    def convert_to_sql(self) -> bool:
        """Convert CSV to SQL database"""
        try:
            db_uri, table_name = convert_csv_to_sqlite(str(self.csv_file), self.db_path, self.table_name)
            return True
        except Exception as e:
            print(f"Error converting CSV to SQL: {str(e)}")
            return False
    
    def query_data(self, sql_query: str) -> List[Dict]:
        """Query data from SQL"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql_query)
            result = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in result]
        except Exception as e:
            print(f"Error querying data: {str(e)}")
            return []