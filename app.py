#supabase working 
#google auth not working 
#version 1 final(deployable)

from flask_cors import CORS
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from authlib.integrations.flask_client import OAuth
import os
import json
import nltk
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import nltk.data
import time
import requests
import urllib.parse
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Supabase Configuration
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Get frontend URL from environment variables or use localhost for development
FRONTEND_URL = os.environ.get('FRONTEND_URL', 'http://localhost:3000')

# Initialize OAuth
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id='921975217993-kka03tvb6ogs8nikfc119c3vc1i50tfk.apps.googleusercontent.com',
    client_secret='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',
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    client_kwargs={'scope': 'openid email profile',
                   'redirect_uri': f"{FRONTEND_URL}/login/google/authorize"},
)

# Initialize the RAG system
API_KEY = "AIzaSyDt6dT2xd1xwMwEOnrwU37Ldks6MvUGWU0"

# Translation function using direct API calls
def translate_text(text, dest_language='hi', src_language='en'):
    """Simple translation function using Google Translate API directly"""
    try:
        encoded_text = urllib.parse.quote(text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={src_language}&tl={dest_language}&dt=t&q={encoded_text}"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            translation_data = response.json()
            if translation_data and len(translation_data) > 0:
                return translation_data[0][0][0]
        return text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# Language codes and their names
LANGUAGES = {
    'en': 'English', 'hi': 'Hindi', 'bn': 'Bengali', 'te': 'Telugu', 'mr': 'Marathi',
    'ta': 'Tamil', 'gu': 'Gujarati', 'kn': 'Kannada', 'ml': 'Malayalam', 'pa': 'Punjabi', 'or': 'Odia'
}

class PDFProcessor:
    """Simple PDF text extractor"""
    def extract_text(self, pdf_path: str) -> str:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text

class TextChunker:
    """Splits text into chunks"""
    def __init__(self, chunk_size: int = 500):  # Reduced chunk size for memory
        self.chunk_size = chunk_size
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def chunk_text(self, text: str) -> list:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

class RAGSystem:
    """RAG system with memory optimization"""
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = API_KEY
        
        # Lazy loading components
        self.pdf_processor = None
        self.chunker = None
        self.embedding_model = None
        self.llm = None
        self.index = None
        self.chunks = []
        self.is_trained = False
        
        # Storage setup
        self.storage_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")
        os.makedirs(self.storage_dir, exist_ok=True)
        self.index_path = os.path.join(self.storage_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(self.storage_dir, "chunks_metadata.json")
        
        # Load or create index
        self.load_or_create_index()
        
        # Initialize API key only
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            print(f"Warning: genai init failed: {e}")
    
    def load_pdf_processor(self):
        """Lazy load PDF processor"""
        if self.pdf_processor is None:
            self.pdf_processor = PDFProcessor()
    
    def load_chunker(self):
        """Lazy load chunker"""
        if self.chunker is None:
            self.chunker = TextChunker()
    
    def load_embedding_model(self):
        """Lazy load embedding model - memory intensive"""
        if self.embedding_model is None:
            print("Loading sentence transformer model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Smaller model
    
    def load_llm(self):
        """Lazy load LLM"""
        if self.llm is None:
            self.llm = genai.GenerativeModel('gemini-2.0-flash')
    
    def load_or_create_index(self):
        """Load existing index and chunks or create new ones"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r') as f:
                    self.chunks = json.load(f)
                print(f"Loaded existing index with {self.index.ntotal} vectors")
            else:
                # Create efficient index type
                dimension = 384
                nlist = 50  # Reduced clusters for memory
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                self.chunks = []
                print("Created new efficient FAISS index")
        except Exception as e:
            print(f"Error loading index: {e}. Creating new one.")
            dimension = 384
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 50)
            self.chunks = []
    
    def save_index_and_chunks(self):
        """Save the FAISS index and chunks metadata"""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.chunks, f)
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def process_pdf(self, pdf_path: str) -> str:
        """Process PDF and store chunks with memory optimization"""
        try:
            self.load_pdf_processor()
            self.load_chunker()
            self.load_embedding_model()
            
            pdf_name = os.path.basename(pdf_path)
            if any(pdf_name in chunk.get('source', '') for chunk in self.chunks):
                return f"PDF {pdf_name} was already processed."
            
            # Process in smaller batches
            text = self.pdf_processor.extract_text(pdf_path)
            new_chunks = self.chunker.chunk_text(text)
            
            # Process chunks in batches to reduce memory peaks
            batch_size = 5
            for i in range(0, len(new_chunks), batch_size):
                batch_chunks = new_chunks[i:i+batch_size]
                embeddings = self.embedding_model.encode(batch_chunks)
                
                # Train index if needed
                if not self.is_trained and len(embeddings) >= 50:
                    self.index.train(embeddings.astype(np.float32))
                    self.is_trained = True
                
                # Add to index if trained
                if self.is_trained:
                    self.index.add(embeddings.astype(np.float32))
                
                # Add metadata
                chunk_metadata = [{'text': chunk, 'source': pdf_name} for chunk in batch_chunks]
                self.chunks.extend(chunk_metadata)
            
            self.save_index_and_chunks()
            return f"Successfully processed {len(new_chunks)} chunks from {pdf_name}"
            
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    
    def query(self, question: str, conversation_history=None, language='en', top_k: int = 3) -> str:
        """Query the system with conversation history and language support"""
        try:
            self.load_embedding_model()
            self.load_llm()
            
            # Get relevant chunks
            query_embedding = self.embedding_model.encode([question])
            scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
            
            # Get context from relevant chunks
            context = " ".join([self.chunks[i]['text'] for i in indices[0] if i < len(self.chunks)])
            
            # Format conversation history
            conversation_context = ""
            if conversation_history:
                conversation_context = "Previous conversation:\n"
                for message in conversation_history:
                    role = "User" if message.get('role') == 'user' else "DrLAW"
                    conversation_context += f"{role}: {message.get('content')}\n"
            
            # Generate answer using Gemini
            output_language = LANGUAGES.get(language, 'English')
            prompt = f"""You are DrLAW, a legal AI advisor. Your task is to provide detailed legal advice based on the following context:

CONTEXT FROM LEGAL DOCUMENTS:
{context}

{conversation_context}

Current Question: {question}

Format your response with clear sections. Use HTML formatting. Include:
1. Brief greeting and introduction
2. Clear, concise answer (200-300 words)
3. Detailed explanation with HTML tables for:
   - Legal Roadmap
   - Required Documentation  
   - Applicable Laws
4. Brief conclusion

Response should be in {output_language}."""

            response = self.llm.generate_content(prompt)
            answer_text = response.text
            
            # Translate if needed
            if language != 'en':
                answer_text = translate_text(answer_text, language, 'en')
            
            return answer_text
        except Exception as e:
            return f"Error generating answer: {str(e)}"

# Initialize RAG system with lazy loading
rag = RAGSystem()

# Process PDFs with memory awareness
def initialize_rag_system():
    """Initialize RAG system gradually to avoid memory peaks"""
    print("Initializing RAG system...")
    
    pdf_files = ["constitution.pdf", "1.pdf", "2.pdf","3.pdf","4.pdf","5.pdf","6.pdf",
                 "7.pdf","8.pdf","9.pdf","10.pdf","11.pdf","12.pdf","13.pdf","14.pdf",
                 "15.pdf","16.pdf","17.pdf","18.pdf","19.pdf","20.pdf","21.pdf","22.pdf",
                 "23.pdf","24.pdf","25.pdf","26.pdf","27.pdf","28.pdf","29.pdf","30.pdf"]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(script_dir, pdf_file)
        if os.path.exists(pdf_path):
            print(f"Processing {pdf_file}...")
            result = rag.process_pdf(pdf_path)
            print(result)
        else:
            print(f"PDF not found: {pdf_file}")

# Database helper functions
def get_user_by_email(email):
    try:
        response = supabase.table("users").select("*").eq("email", email).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"Error getting user by email: {e}")
        return None

def create_user(username, email, password_hash=None, google_id=None):
    try:
        user_data = {"username": username, "email": email}
        if password_hash:
            user_data["password_hash"] = password_hash
        if google_id:
            user_data["google_id"] = google_id
            
        response = supabase.table("users").insert(user_data).execute()
        return response.data[0]["user_id"]
    except Exception as e:
        print(f"Error creating user: {e}")
        return None

def save_chat(user_id, question, answer):
    try:
        chat_data = {"user_id": user_id, "question": question, "answer": answer}
        supabase.table("chats").insert(chat_data).execute()
    except Exception as e:
        print(f"Error saving chat: {e}")

def get_chat_history(user_id):
    try:
        response = supabase.table("chats").select("*").eq("user_id", user_id).order("timestamp", desc=True).execute()
        return response.data
    except Exception as e:
        print(f"Error getting chat history: {e}")
        return []

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Flask routes
@app.route('/')
def index():
    if 'user_id' in session:
        return render_template('index.html')
    return render_template('front.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = get_user_by_email(email)
        
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['user_id']
            session['username'] = user['username']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'danger')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user_id' in session:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if get_user_by_email(email):
            flash('Email already registered', 'danger')
            return redirect(url_for('signup'))
        
        try:
            password_hash = generate_password_hash(password)
            user_id = create_user(username, email, password_hash)
            session['user_id'] = user_id
            session['username'] = username
            flash('Account created successfully!', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            flash('Error creating account', 'danger')
    
    return render_template('front.html')

@app.route('/login/google')
def google_login():
    redirect_uri = url_for('google_authorize', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/login/google/authorize')
def google_authorize():
    try:
        token = google.authorize_access_token()
        user_info = google.get('userinfo').json()
        
        response = supabase.table("users").select("*").eq("google_id", user_info['id']).execute()
        user = response.data[0] if response.data else None
        
        if not user:
            response = supabase.table("users").select("*").eq("email", user_info['email']).execute()
            user = response.data[0] if response.data else None
            
            if user:
                supabase.table("users").update({"google_id": user_info['id']}).eq("user_id", user['user_id']).execute()
            else:
                username = user_info.get('name', user_info['email'].split('@')[0])
                user_id = create_user(username=username, email=user_info['email'], google_id=user_info['id'])
                user = {'user_id': user_id, 'username': username}
        
        session['user_id'] = user['user_id']
        session['username'] = user['username']
        flash('Logged in with Google successfully!', 'success')
        return redirect(url_for('index'))
    
    except Exception as e:
        flash('Error logging in with Google', 'danger')
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

@app.route('/ask', methods=['POST'])
@login_required
def ask():
    data = request.json
    question = data.get('question')
    language = data.get('language', 'en')
    conversation_history = data.get('conversation_history', [])
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    answer = rag.query(question, conversation_history, language)
    save_chat(session['user_id'], question, answer)
    
    return jsonify({'question': question, 'answer': answer, 'language': language})

@app.route('/chat/history')
@login_required
def chat_history():
    history = get_chat_history(session['user_id'])
    return jsonify([dict(row) for row in history])

# Initialize RAG system on first request
@app.before_first_request
def initialize_app():
    initialize_rag_system()

# fix main guard
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)  # debug=False for production



    