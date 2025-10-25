"""
models.py - OOP Models for DocuMind
Contains all class definitions for Document Q&A and CSV/SQL features
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Optional
import os


# ============= USER HIERARCHY =============

class User:
    """Base User class"""
    
    def __init__(self, user_id: str, email: str, name: str, password: str):
        self.user_id = user_id
        self.email = email
        self.name = name
        self.password = password
        self.created_at = datetime.now()
    
    def login(self, email: str, password: str) -> bool:
        """Verify user login credentials"""
        return self.email == email and self.password == password
    
    def logout(self) -> None:
        """Log out user"""
        print(f"User {self.name} logged out")
    
    def update_profile(self, name: str) -> bool:
        """Update user profile"""
        self.name = name
        return True
    
    def change_password(self, old_password: str, new_password: str) -> bool:
        """Change user password"""
        if self.password == old_password:
            self.password = new_password
            return True
        return False


class GuestUser(User):
    """Guest User - temporary session"""
    
    def __init__(self, user_id: str):
        super().__init__(user_id, "guest@example.com", "Guest", "guest123")
        self.session_token = user_id
        self.expiry_time = datetime.now()
    
    def register(self, email: str, name: str, password: str) -> 'RegisteredUser':
        """Convert guest to registered user"""
        return RegisteredUser(self.user_id, email, name, password)
    
    def validate_token(self) -> bool:
        """Validate session token"""
        return True
    
    def ask_query(self, query_text: str) -> 'Response':
        """Ask a query (limited for guests)"""
        pass


class RegisteredUser(User):
    """Registered User - full access"""
    
    def __init__(self, user_id: str, email: str, name: str, password: str):
        super().__init__(user_id, email, name, password)
        self.documents_count = 0
        self.subscription = "free"
        self.documents: List['Document'] = []
    
    def upload_document(self, file: 'File', doc_type: str) -> 'Document':
        """Upload a document"""
        if doc_type.lower() == "pdf":
            doc = PDFDocument(f"doc_{self.documents_count}", file.name, self.user_id)
        elif doc_type.lower() == "csv":
            doc = CSVDocument(f"doc_{self.documents_count}", file.name, self.user_id)
        elif doc_type.lower() == "docx":
            doc = WordDocument(f"doc_{self.documents_count}", file.name, self.user_id)
        else:
            doc = TextDocument(f"doc_{self.documents_count}", file.name, self.user_id)
        
        self.documents.append(doc)
        self.documents_count += 1
        return doc
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document"""
        self.documents = [d for d in self.documents if d.doc_id != doc_id]
        return True
    
    def get_documents(self) -> List['Document']:
        """Get all documents"""
        return self.documents
    
    def ask_query(self, doc_id: str, query_text: str) -> 'Response':
        """Ask a query about a document"""
        pass
    
    def view_chat_history(self) -> List[Dict]:
        """View chat history"""
        pass


class Admin(User):
    """Admin User - system management"""
    
    def __init__(self, user_id: str, email: str, name: str):
        super().__init__(user_id, email, name, "admin123")
        self.admin_level = 2
        self.permissions = ["manage_users", "manage_documents", "view_logs"]
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user account"""
        print(f"User {user_id} deleted by admin")
        return True
    
    def suspend_user(self, user_id: str) -> bool:
        """Suspend a user"""
        print(f"User {user_id} suspended")
        return True
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document"""
        print(f"Document {doc_id} deleted by admin")
        return True
    
    def view_system_logs(self) -> List[str]:
        """View system logs"""
        return ["Log 1", "Log 2", "Log 3"]


# ============= DOCUMENT HIERARCHY =============

class Document(ABC):
    """Abstract Document class"""
    
    def __init__(self, doc_id: str, title: str, uploaded_by: str):
        self.doc_id = doc_id
        self.title = title
        self.file_type = "document"
        self.uploaded_by = uploaded_by
        self.uploaded_at = datetime.now()
        self.file_size = 0
        self.content = ""
    
    @abstractmethod
    def parse(self) -> List[str]:
        """Parse document and extract chunks"""
        pass
    
    def get_content(self) -> str:
        """Get document content"""
        return self.content
    
    def delete(self) -> bool:
        """Delete document"""
        return True
    
    def get_metadata(self) -> Dict[str, str]:
        """Get document metadata"""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "file_type": self.file_type,
            "uploaded_by": self.uploaded_by,
            "uploaded_at": str(self.uploaded_at)
        }


class PDFDocument(Document):
    """PDF Document"""
    
    def __init__(self, doc_id: str, title: str, uploaded_by: str):
        super().__init__(doc_id, title, uploaded_by)
        self.file_type = "pdf"
        self.page_count = 0
        self.author = "Unknown"
    
    def parse(self) -> List[str]:
        """Parse PDF and extract text chunks"""
        return ["PDF chunk 1", "PDF chunk 2", "PDF chunk 3"]
    
    def extract_text(self) -> str:
        """Extract text from PDF"""
        return "Extracted text from PDF"
    
    def get_page_count(self) -> int:
        """Get number of pages"""
        return self.page_count
    
    def search_keyword(self, keyword: str) -> List[int]:
        """Search keyword in PDF"""
        return [0, 2, 5]


class CSVDocument(Document):
    """CSV Document"""
    
    def __init__(self, doc_id: str, title: str, uploaded_by: str):
        super().__init__(doc_id, title, uploaded_by)
        self.file_type = "csv"
        self.table_name = "data"
        self.row_count = 0
        self.column_count = 0
        self.columns: List[str] = []
    
    def parse(self) -> List[str]:
        """Parse CSV and extract rows"""
        return ["row1", "row2", "row3"]
    
    def convert_to_sql(self) -> str:
        """Convert CSV to SQL"""
        return "CREATE TABLE data (id INT, name TEXT)"
    
    def validate_data(self) -> bool:
        """Validate CSV data"""
        return True
    
    def get_table_schema(self) -> Dict[str, str]:
        """Get table schema"""
        return {"id": "INTEGER", "name": "TEXT"}


class WordDocument(Document):
    """Word Document"""
    
    def __init__(self, doc_id: str, title: str, uploaded_by: str):
        super().__init__(doc_id, title, uploaded_by)
        self.file_type = "docx"
        self.author = "Unknown"
        self.revision_count = 0
    
    def parse(self) -> List[str]:
        """Parse DOCX and extract text chunks"""
        return ["Word chunk 1", "Word chunk 2"]
    
    def extract_tables(self) -> List[Dict]:
        """Extract tables from document"""
        return [{"table": 1, "rows": 5}]
    
    def extract_text(self) -> str:
        """Extract text from document"""
        return "Extracted text from Word"
    
    def get_author(self) -> str:
        """Get document author"""
        return self.author


class TextDocument(Document):
    """Text Document"""
    
    def __init__(self, doc_id: str, title: str, uploaded_by: str):
        super().__init__(doc_id, title, uploaded_by)
        self.file_type = "txt"
        self.encoding = "utf-8"
        self.line_count = 0
    
    def parse(self) -> List[str]:
        """Parse TXT and extract lines"""
        return ["line 1", "line 2", "line 3"]
    
    def get_line_count(self) -> int:
        """Get number of lines"""
        return self.line_count
    
    def get_line(self, line_num: int) -> str:
        """Get specific line"""
        return f"Line {line_num}"


# ============= QUERY & RESPONSE =============

class Query:
    """Query class"""
    
    def __init__(self, query_id: str, query_text: str, user_id: str, doc_id: Optional[str] = None):
        self.query_id = query_id
        self.query_text = query_text
        self.user_id = user_id
        self.doc_id = doc_id
        self.created_at = datetime.now()
        self.status = "pending"
    
    def execute(self) -> 'Response':
        """Execute query"""
        pass
    
    def get_query_text(self) -> str:
        """Get query text"""
        return self.query_text
    
    def validate(self) -> bool:
        """Validate query"""
        return len(self.query_text) > 0
    
    def set_status(self, status: str) -> None:
        """Set query status"""
        self.status = status


class SQLQuery(Query):
    """SQL Query class"""
    
    def __init__(self, query_id: str, query_text: str, user_id: str, doc_id: str):
        super().__init__(query_id, query_text, user_id, doc_id)
        self.sql_text = ""
        self.is_auto_generated = False
    
    def execute(self) -> 'Response':
        """Execute SQL query"""
        pass
    
    def execute_sql(self, connection) -> List[Dict]:
        """Execute SQL on database"""
        return [{"id": 1, "name": "result"}]
    
    def generate_from_nlp(self, nlp_text: str) -> str:
        """Generate SQL from natural language"""
        return "SELECT * FROM data"
    
    def get_sql_text(self) -> str:
        """Get SQL text"""
        return self.sql_text


class Response:
    """Response class"""
    
    def __init__(self, response_id: str, response_text: str, query_id: str):
        self.response_id = response_id
        self.response_text = response_text
        self.query_id = query_id
        self.sources: List[str] = []
        self.generated_at = datetime.now()
        self.confidence = 0.85
    
    def get_response_text(self) -> str:
        """Get response text"""
        return self.response_text
    
    def get_sources(self) -> List[str]:
        """Get source chunks"""
        return self.sources
    
    def get_confidence(self) -> float:
        """Get confidence score"""
        return self.confidence
    
    def generate_visualization(self) -> Optional['Visualization']:
        """Generate visualization from response"""
        return None
    
    def export_as_markdown(self) -> str:
        """Export response as markdown"""
        return f"# Response\n\n{self.response_text}"


class Visualization:
    """Visualization class"""
    
    def __init__(self, viz_id: str, response_id: str, viz_type: str):
        self.viz_id = viz_id
        self.response_id = response_id
        self.viz_type = viz_type  # BarChart, LineChart, etc.
        self.title = ""
        self.data = []
    
    def get_viz_type(self) -> str:
        """Get visualization type"""
        return self.viz_type
    
    def generate_chart(self):
        """Generate chart"""
        pass
    
    def export_as_image(self, format: str):
        """Export as image"""
        pass
    
    def export_as_json(self) -> str:
        """Export as JSON"""
        return str(self.data)
    
    def update_data(self, new_data: List[Dict]) -> None:
        """Update visualization data"""
        self.data = new_data


# ============= SESSION & CHAT =============

class ChatMessage:
    """Chat Message class"""
    
    def __init__(self, message_id: str, session_id: str, sender: str, content: str):
        self.message_id = message_id
        self.session_id = session_id
        self.sender = sender  # "user" or "assistant"
        self.content = content
        self.timestamp = datetime.now()
    
    def get_sender(self) -> str:
        """Get sender"""
        return self.sender
    
    def get_content(self) -> str:
        """Get message content"""
        return self.content
    
    def get_timestamp(self) -> datetime:
        """Get timestamp"""
        return self.timestamp
    
    def edit(self, new_content: str) -> bool:
        """Edit message"""
        self.content = new_content
        return True
    
    def delete(self) -> bool:
        """Delete message"""
        return True


class Session:
    """Session class"""
    
    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self.start_time = datetime.now()
        self.end_time = None
        self.is_active = True
        self.messages: List[ChatMessage] = []
        self.queries: List[Query] = []
        self.responses: List[Response] = []
    
    def add_message(self, message: ChatMessage) -> None:
        """Add message to session"""
        self.messages.append(message)
    
    def add_query_response(self, query: Query, response: Response) -> None:
        """Add query and response"""
        self.queries.append(query)
        self.responses.append(response)
    
    def get_history(self) -> List[ChatMessage]:
        """Get chat history"""
        return self.messages
    
    def end_session(self) -> None:
        """End session"""
        self.end_time = datetime.now()
        self.is_active = False
    
    def get_duration(self) -> int:
        """Get session duration in seconds"""
        if self.end_time:
            return int((self.end_time - self.start_time).total_seconds())
        return int((datetime.now() - self.start_time).total_seconds())
    
    def export_transcript(self) -> str:
        """Export session transcript"""
        transcript = f"Session {self.session_id}\n"
        for msg in self.messages:
            transcript += f"{msg.sender}: {msg.content}\n"
        return transcript