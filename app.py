import os
import glob
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
import pathway as pw
import asyncio
from typing import List, Optional
import shutil
import requests
import json
import logging

# -------------------------------
# Configure logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Initialize FastAPI app with CORS
# -------------------------------
app = FastAPI(title="RAG API with Pathway + Ollama")

# Enhanced CORS configuration
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://0.0.0.0:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    # Add any other origins you need
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Ollama Configuration
# -------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")  # Configurable model

# -------------------------------
# Helper Functions for Robust API Calls
# -------------------------------
def make_ollama_request(url_suffix: str, payload: Optional[dict] = None, method: str = "GET", timeout: int = 30, max_retries: int = 3):
    """Make a robust request to Ollama API with retry logic"""
    url = f"{OLLAMA_HOST}{url_suffix}"
    
    for attempt in range(max_retries):
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=payload, timeout=timeout)
            else:
                response = requests.post(url, json=payload, timeout=timeout)
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Ollama connection error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=503, detail=f"Cannot connect to Ollama at {OLLAMA_HOST}. Make sure Ollama is running.")
            time.sleep(1)  # Wait before retrying
            
        except requests.exceptions.Timeout as e:
            logger.warning(f"Ollama request timeout (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=504, detail="Ollama request timeout. The model might be busy or too slow.")
            time.sleep(1)  # Wait before retrying
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"Ollama HTTP error: {e}")
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Ollama model '{OLLAMA_MODEL}' not found. Available models: {get_available_models()}")
            else:
                raise HTTPException(status_code=response.status_code, detail=f"Ollama API error: {str(e)}")
        
        except Exception as e:
            logger.error(f"Unexpected error contacting Ollama: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
            time.sleep(1)  # Wait before retrying

def get_available_models():
    """Get list of available Ollama models"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            return [model['name'] for model in response.json().get('models', [])]
        return []
    except:
        return []

# -------------------------------
# Pathway RAG System
# -------------------------------
class PathwayRAGSystem:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.documents = []
        self.file_paths = []
        self.initialized = False
        
    def load_documents(self):
        """Load all text files from data directory"""
        documents = []
        file_paths = []
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load all text files
        for ext in ['*.txt', '*.pdf']:
            for file_path in glob.glob(os.path.join(self.data_dir, ext)):
                try:
                    if file_path.endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            documents.append(content)
                            file_paths.append(file_path)
                    elif file_path.endswith('.pdf'):
                        text = extract_text_from_pdf(file_path)
                        if text.strip():
                            documents.append(text.strip())
                            file_paths.append(file_path)
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}")
        
        return documents, file_paths
    
    def initialize(self):
        """Initialize the RAG system"""
        try:
            documents, file_paths = self.load_documents()
            if documents:
                self.documents = documents
                self.file_paths = file_paths
                self.initialized = True
                logger.info(f"RAG system initialized with {len(documents)} documents")
            else:
                logger.info("No documents found in data directory")
                self.initialized = True  # Mark as initialized even with no documents
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            self.initialized = False
    
    def search_documents(self, query: str, top_k: int = 3) -> List[str]:
        """Search for relevant documents with better matching"""
        if not self.initialized or not self.documents:
            return ["No documents available for search"]
        
        # Better text-based search with scoring
        query_lower = query.lower()
        scored_docs = []
        
        for i, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            score = 0
            
            # Score based on word matches
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 3:  # Only count meaningful words
                    if word in doc_lower:
                        score += 1
            
            # Higher score for exact phrase matches
            if query_lower in doc_lower:
                score += 5
                
            if score > 0:
                scored_docs.append((score, doc, i))
        
        # Sort by score (highest first)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k documents
        relevant_docs = [doc for score, doc, idx in scored_docs[:top_k]]
        
        # If no matches, return first few documents with a note
        if not relevant_docs:
            relevant_docs = self.documents[:min(top_k, len(self.documents))]
            if relevant_docs:
                relevant_docs[0] = "No specific matches found. Here's general information:\n" + relevant_docs[0]
        
        return relevant_docs

# Initialize RAG system
rag_system = PathwayRAGSystem()

# -------------------------------
# Helper Functions
# -------------------------------
def extract_text_from_pdf(file_path):
    """Extract text from PDF file with improved error handling"""
    try:
        with open(file_path, 'rb') as f:
            pdf = PdfReader(f)
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {e}")
        return ""

async def generate_response_with_rag(query: str, context: List[str]):
    """Generate response using Ollama with RAG context"""
    # Limit context length to prevent overly long prompts
    limited_context = []
    for i, ctx in enumerate(context):
        if len(ctx) > 800:  # Reduced from 1000 to 800
            limited_context.append(f"Context {i+1}:\n{ctx[:800]}... [truncated]")
        else:
            limited_context.append(f"Context {i+1}:\n{ctx}")
    
    context_text = "\n\n".join(limited_context)
    
    prompt = f"""Based on the context below, answer the question concisely.

{context_text}

Question: {query}

Short answer:"""
    
    try:
        # Use our robust request function
        response = make_ollama_request(
            "/api/generate",
            payload={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 200  # Reduced from 500 to 200
                }
            },
            method="POST",
            timeout=70,
            max_retries=2
        )
        
        return response.json()["response"]
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_response_with_rag: {e}")
        # Fallback: return the most relevant context
        return f"Based on the documents, here's what I found: {context[0][:500]}..." if context else "No relevant information found in documents."

# -------------------------------
# API Endpoints
# -------------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup and verify Ollama connection"""
    rag_system.initialize()
    
    # Test Ollama connection on startup
    try:
        response = make_ollama_request("/api/tags", timeout=10, max_retries=2)
        models = [model['name'] for model in response.json().get('models', [])]
        logger.info(f"Connected to Ollama. Available models: {models}")
        
        # Check if configured model is available
        if OLLAMA_MODEL not in models:
            logger.warning(f"Configured model '{OLLAMA_MODEL}' not found in available models: {models}")
    except Exception as e:
        logger.warning(f"Ollama not available at startup: {e}")

@app.get("/health")
async def health():
    # Test Ollama connection
    ollama_healthy = False
    ollama_models = []
    
    try:
        response = make_ollama_request("/api/tags", timeout=5, max_retries=1)
        ollama_healthy = True
        ollama_models = [model['name'] for model in response.json().get('models', [])]
    except Exception as e:
        ollama_healthy = False
    
    return {
        "status": "ok", 
        "pathway_version": pw.__version__,
        "rag_initialized": rag_system.initialized,
        "document_count": len(rag_system.documents) if rag_system.initialized else 0,
        "ollama_connected": ollama_healthy,
        "ollama_model": OLLAMA_MODEL,
        "available_models": ollama_models,
        "ollama_host": OLLAMA_HOST
    }

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file to the data directory"""
    try:
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Validate file type
        if not (file.filename.endswith('.txt') or file.filename.endswith('.pdf')):
            raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported")
        
        # Save file to data directory
        file_path = os.path.join("data", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Reinitialize RAG system with new file
        rag_system.initialize()
        
        return {
            "message": f"File {file.filename} uploaded successfully",
            "rag_reloaded": rag_system.initialized
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/query/")
async def query_rag(
    question: str = Query(..., description="The question to answer"),
    top_k: int = Query(3, description="Number of context documents to use")
):
    """Query the RAG system with a question"""
    try:
        if not rag_system.initialized:
            rag_system.initialize()
            
        if not rag_system.documents:
            raise HTTPException(status_code=400, detail="No documents available. Please upload documents first.")
        
        # Search for relevant documents
        relevant_docs = rag_system.search_documents(question, top_k)
        
        # Generate response using Ollama
        response = await generate_response_with_rag(question, relevant_docs)
        
        return {
            "question": question,
            "context_documents_used": len(relevant_docs),
            "response": response,
            "rag_system": "Pathway + Ollama",
            "model_used": OLLAMA_MODEL
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error in query_rag: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/documents/")
async def list_documents():
    """List all documents in the RAG system"""
    try:
        if not rag_system.initialized:
            rag_system.initialize()
            
        return {
            "document_count": len(rag_system.documents) if rag_system.initialized else 0,
            "documents": [
                {
                    "path": rag_system.file_paths[i],
                    "preview": doc[:200] + "..." if len(doc) > 200 else doc
                }
                for i, doc in enumerate(rag_system.documents)
            ] if rag_system.initialized else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.post("/reload/")
async def reload_rag():
    """Reload the RAG system with current documents"""
    try:
        rag_system.initialize()
        return {
            "message": "RAG system reloaded",
            "document_count": len(rag_system.documents) if rag_system.initialized else 0,
            "initialized": rag_system.initialized
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading RAG system: {str(e)}")

@app.get("/ollama-models/")
async def list_ollama_models():
    """List available Ollama models"""
    try:
        response = make_ollama_request("/api/tags", timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e), "host": OLLAMA_HOST}

@app.get("/ollama-test/")
async def test_ollama():
    """Test Ollama connection with a simple prompt"""
    try:
        response = make_ollama_request(
            "/api/generate",
            payload={
                "model": OLLAMA_MODEL,
                "prompt": "Hello, how are you?",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 50
                }
            },
            method="POST",
            timeout=15
        )
        
        return {
            "status": "success",
            "response": response.json()["response"],
            "model": OLLAMA_MODEL
        }
    except HTTPException as e:
        # Re-raise HTTP exceptions to maintain status code
        raise e
    except Exception as e:
        return {"status": "error", "message": str(e)}