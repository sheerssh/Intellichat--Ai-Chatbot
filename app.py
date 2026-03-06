from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
from datetime import datetime
import os
import shutil

# LangChain + ChromaDB imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader

app = Flask(__name__)
CORS(app)

# ── CONFIG ──────────────────────────────────────────────
GROQ_API_KEY = ''
UPLOAD_FOLDER = 'uploaded_docs'
CHROMA_DB_DIR = 'chroma_db'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── GLOBALS ─────────────────────────────────────────────
conversation_history = []
vectorstore          = None   # ChromaDB instance
doc_name             = None   # Name of currently loaded document

# ── EMBEDDINGS MODEL (runs locally, no API key needed) ──
print("⏳ Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",   # small, fast, free
    model_kwargs={"device": "cpu"}
)
print("✅ Embedding model ready.")


# ════════════════════════════════════════════════════════
#  DOCUMENT LOADING & INDEXING
# ════════════════════════════════════════════════════════

def load_and_index_document(filepath, filename):
    """Load a PDF or TXT file, split into chunks, store in ChromaDB."""
    global vectorstore, doc_name

    ext = filename.rsplit('.', 1)[-1].lower()

    # Load document
    if ext == 'pdf':
        loader = PyPDFLoader(filepath)
    elif ext in ['txt', 'md']:
        loader = TextLoader(filepath, encoding='utf-8')
    else:
        return False, "Unsupported file type. Use PDF or TXT."

    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        return False, "Could not extract text from the document."

    # Clear old ChromaDB and create new one
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    doc_name = filename
    print(f"✅ Indexed '{filename}' → {len(chunks)} chunks stored in ChromaDB.")
    return True, f"Document '{filename}' indexed successfully with {len(chunks)} chunks."


def get_relevant_context(query, k=4):
    """Retrieve top-k relevant chunks from ChromaDB for the query."""
    if vectorstore is None:
        return None
    results = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in results])
    return context


# ════════════════════════════════════════════════════════
#  GROQ API
# ════════════════════════════════════════════════════════

def query_groq(message, context=None):
    """Query Groq API, optionally with document context."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Build system prompt
    if context:
        system_prompt = f"""You are IntelliChat, a helpful AI assistant.
The user has uploaded a document. Use the context below to answer their question accurately.
If the answer is not in the context, say so honestly.

--- DOCUMENT CONTEXT ---
{context}
--- END OF CONTEXT ---

Answer in clear, concise bullet points or short paragraphs."""
    else:
        system_prompt = """You are IntelliChat, a helpful AI assistant.
Answer in short, clear bullet points or small paragraphs. Keep responses simple."""

    messages = [{"role": "system", "content": system_prompt}]

    # Add recent conversation history
    recent = [m for m in conversation_history[-6:] if m['role'] in ['user', 'assistant']]
    for msg in recent:
        messages.append({"role": msg['role'], "content": msg['content']})

    messages.append({"role": "user", "content": message})

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 700
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            print(f"Groq Error: {response.status_code} - {response.text}")
            return None
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Groq API Error: {e}")
        return None


def offline_response(message):
    """Fallback offline responses."""
    message_lower = message.lower()
    if any(w in message_lower for w in ['hello','hi','hey']):
        return "Hello! How can I help you today?"
    if 'your name' in message_lower or 'who are you' in message_lower:
        return "I'm IntelliChat, your AI assistant!"
    if 'time' in message_lower:
        return f"The current time is {datetime.now().strftime('%I:%M %p')}."
    if 'date' in message_lower or 'today' in message_lower:
        return f"Today is {datetime.now().strftime('%B %d, %Y')}."
    if any(w in message_lower for w in ['thank','thanks']):
        return "You're welcome!"
    if any(w in message_lower for w in ['bye','goodbye']):
        return "Goodbye! Have a great day!"
    return "I'm in offline mode right now. Please check your Groq API key."


# ════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data         = request.json
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Add user message to history
        conversation_history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })

        # Try to get relevant context from ChromaDB
        context = get_relevant_context(user_message) if vectorstore else None

        # Get bot response
        bot_message = query_groq(user_message, context=context)

        if bot_message is None:
            bot_message = offline_response(user_message)

        # Tag response if it used document context
        source_tag = f"\n\n📄 *Answered using: {doc_name}*" if context and doc_name else ""

        # Add to history
        conversation_history.append({
            'role': 'assistant',
            'content': bot_message,
            'timestamp': datetime.now().isoformat()
        })

        return jsonify({
            'response': bot_message + source_tag,
            'used_document': context is not None,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error in /chat: {e}")
        return jsonify({'error': 'An error occurred. Please try again.'}), 500


@app.route('/upload_doc', methods=['POST'])
def upload_doc():
    """Upload and index a document into ChromaDB."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        allowed = {'pdf', 'txt', 'md'}
        ext = file.filename.rsplit('.', 1)[-1].lower()
        if ext not in allowed:
            return jsonify({'error': 'Only PDF, TXT, and MD files are supported'}), 400

        # Save the file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Index it
        success, message = load_and_index_document(filepath, file.filename)

        if success:
            return jsonify({'message': message, 'filename': file.filename})
        else:
            return jsonify({'error': message}), 400

    except Exception as e:
        print(f"Error in /upload_doc: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/clear_doc', methods=['POST'])
def clear_doc():
    """Remove the currently loaded document from ChromaDB."""
    global vectorstore, doc_name
    vectorstore = None
    doc_name    = None
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
    return jsonify({'message': 'Document cleared successfully'})


@app.route('/doc_status', methods=['GET'])
def doc_status():
    """Return whether a document is currently loaded."""
    return jsonify({
        'loaded': vectorstore is not None,
        'filename': doc_name
    })


@app.route('/clear', methods=['POST'])
def clear_history():
    global conversation_history
    conversation_history = []
    return jsonify({'message': 'Conversation cleared'})


@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({'history': conversation_history})


# ════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("🚀 Starting IntelliChat Server...")
    print("✅ Groq API Mode: Enabled")
    print("🧠 RAG Mode: LangChain + ChromaDB")
    print("🌐 Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)