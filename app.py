from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
from datetime import datetime
import os
import shutil
import time
import uuid

# LangChain + ChromaDB imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader

app = Flask(__name__)
CORS(app)

# ── CONFIG ──────────────────────────────────────────────
GROQ_API_KEY  = ''
UPLOAD_FOLDER = 'uploaded_docs'
CHROMA_DB_DIR = 'chroma_db'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── GLOBALS ─────────────────────────────────────────────
# Multi-session storage
# sessions = { session_id: { 'title': str, 'history': [], 'created': str } }
sessions        = {}
active_session  = None   # current session id

vectorstore     = None
doc_name        = None
response_times  = []

# ── EMBEDDINGS ──────────────────────────────────────────
print("⏳ Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
print("✅ Embedding model ready.")


# ════════════════════════════════════════════════════════
#  SESSION HELPERS
# ════════════════════════════════════════════════════════

def create_session(title="New Chat"):
    sid = str(uuid.uuid4())[:8]
    sessions[sid] = {
        'id':      sid,
        'title':   title,
        'history': [],
        'created': datetime.now().isoformat()
    }
    return sid

def get_active_history():
    if active_session and active_session in sessions:
        return sessions[active_session]['history']
    return []

def auto_title(message):
    """Use first 40 chars of first user message as title."""
    return message[:40] + ("..." if len(message) > 40 else "")


# ════════════════════════════════════════════════════════
#  DOCUMENT LOADING & INDEXING
# ════════════════════════════════════════════════════════

def load_and_index_document(filepath, filename):
    global vectorstore, doc_name
    ext = filename.rsplit('.', 1)[-1].lower()

    if ext == 'pdf':
        loader = PyPDFLoader(filepath)
    elif ext in ['txt', 'md']:
        loader = TextLoader(filepath, encoding='utf-8')
    else:
        return False, "Unsupported file type."

    documents = loader.load()
    splitter  = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks    = splitter.split_documents(documents)

    if not chunks:
        return False, "Could not extract text from document."

    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    doc_name = filename
    print(f"✅ Indexed '{filename}' → {len(chunks)} chunks.")
    return True, f"Document '{filename}' indexed with {len(chunks)} chunks."


def get_relevant_context(query, k=4):
    if vectorstore is None:
        return None
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])


# ════════════════════════════════════════════════════════
#  GROQ API
# ════════════════════════════════════════════════════════

def query_groq(message, history, context=None):
    url     = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    if context:
        system_prompt = f"""You are IntelliChat, a helpful AI assistant.
Use the document context below to answer the user's question accurately.
If the answer is not in the context, say so honestly.

--- DOCUMENT CONTEXT ---
{context}
--- END OF CONTEXT ---

Answer in clear, concise bullet points or short paragraphs."""
    else:
        system_prompt = "You are IntelliChat, a helpful AI assistant. Answer in short, clear bullet points or small paragraphs."

    messages = [{"role": "system", "content": system_prompt}]
    recent   = [m for m in history[-6:] if m['role'] in ['user', 'assistant']]
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
            return None
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Groq API Error: {e}")
        return None


def offline_response(message):
    message_lower = message.lower()

    # ── Greetings ──
    if any(w in message_lower for w in ['hello','hi','hey','howdy','greetings','sup']):
        return "Hello! 👋 I'm IntelliChat running in **offline mode** right now. I can answer some basic questions, but for full AI responses please check your Groq API key."

    # ── How are you ──
    if any(p in message_lower for p in ['how are you','how do you do','you doing','whats up','what\'s up']):
        return "I'm doing fine, thanks for asking! 😊 Note: I'm currently in **offline mode** — my responses are limited."

    # ── Name / identity ──
    if any(p in message_lower for p in ['your name','who are you','what are you','introduce']):
        return "I'm **IntelliChat**, your AI assistant! I'm currently running in offline mode, so my capabilities are limited right now."

    # ── Time ──
    if 'time' in message_lower and 'what' in message_lower:
        return f"🕐 The current time is **{datetime.now().strftime('%I:%M %p')}**."

    # ── Date ──
    if any(w in message_lower for w in ['date','today','day is']):
        return f"📅 Today is **{datetime.now().strftime('%A, %B %d, %Y')}**."

    # ── Day of week ──
    if 'day' in message_lower and any(w in message_lower for w in ['what','which']):
        return f"📅 Today is **{datetime.now().strftime('%A')}**."

    # ── Weather ──
    if 'weather' in message_lower:
        return "🌤️ I don't have access to real-time weather data. Please check a weather website or app for your local forecast!"

    # ── Jokes ──
    if any(w in message_lower for w in ['joke','funny','laugh']):
        import random
        jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
            "Why did the developer go broke? Because he used up all his cache! 💸",
            "What do you call a programmer from Finland? Nerdic! 😄",
            "Why do Java developers wear glasses? Because they don't C#! 👓",
            "A SQL query walks into a bar, walks up to two tables and asks... 'Can I join you?' 😂"
        ]
        return random.choice(jokes)

    # ── Math ──
    if any(w in message_lower for w in ['calculate','what is','compute']) and any(c in message_lower for c in ['+','-','*','/','^','percent','%']):
        try:
            expr = message_lower.replace('what is','').replace('calculate','').replace('compute','').strip()
            expr = expr.replace('^','**').replace('x','*').replace('×','*').replace('÷','/')
            result = eval(expr, {"__builtins__": {}})
            return f"🔢 The answer is **{result}**."
        except:
            return "🔢 I couldn't calculate that. Please check the expression and try again."

    # ── Capitals ──
    if 'capital' in message_lower:
        capitals = {
            'india':'New Delhi','usa':'Washington D.C.','france':'Paris',
            'germany':'Berlin','japan':'Tokyo','china':'Beijing',
            'uk':'London','australia':'Canberra','canada':'Ottawa',
            'brazil':'Brasília','russia':'Moscow','italy':'Rome',
            'spain':'Madrid','mexico':'Mexico City','south korea':'Seoul'
        }
        for country, capital in capitals.items():
            if country in message_lower:
                return f"🏛️ The capital of **{country.title()}** is **{capital}**."
        return "🏛️ I know capitals of many countries! Try asking 'What is the capital of France?'"

    # ── Thanks ──
    if any(w in message_lower for w in ['thank','thanks','appreciate','grateful']):
        return "You're welcome! 😊 Happy to help!"

    # ── Goodbye ──
    if any(w in message_lower for w in ['bye','goodbye','see you','cya','take care','farewell']):
        return "Goodbye! 👋 Have a wonderful day!"

    # ── Help ──
    if any(w in message_lower for w in ['help','what can you do','capabilities','features']):
        return """Here's what I can help with in **offline mode**:
- 🕐 Current time and date
- 🌍 Country capitals
- 🔢 Basic calculations
- 😄 Jokes
- 💬 General conversation

For full AI capabilities, please ensure your **Groq API key** is valid and your internet connection is working."""

    # ── Python ──
    if 'python' in message_lower and any(w in message_lower for w in ['what','tell','explain']):
        return "🐍 **Python** is a popular high-level programming language known for its simple syntax and readability. It's widely used in AI, web development, data science, and automation."

    # ── AI ──
    if any(p in message_lower for p in ['what is ai','what is artificial intelligence','explain ai']):
        return "🤖 **Artificial Intelligence (AI)** is the simulation of human intelligence by computer systems. It includes machine learning, natural language processing, computer vision, and more."

    # ── Default ──
    return f"""⚠️ I'm currently in **offline mode** — the Groq API is unavailable.

I can answer basic questions about:
- Time & date
- Country capitals  
- Simple math
- General conversation

For full responses, please check:
1. Your internet connection
2. Your Groq API key in the `.env` file
3. That Flask server restarted after `.env` changes"""


# ════════════════════════════════════════════════════════
#  ROUTES — SESSIONS
# ════════════════════════════════════════════════════════

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/sessions', methods=['GET'])
def get_sessions():
    """Return all sessions sorted by creation time (newest first)."""
    session_list = sorted(
        sessions.values(),
        key=lambda s: s['created'],
        reverse=True
    )
    # Return lightweight version (no full history)
    return jsonify({
        'sessions': [
            {'id': s['id'], 'title': s['title'], 'created': s['created'],
             'message_count': len(s['history'])}
            for s in session_list
        ],
        'active': active_session
    })


@app.route('/sessions/new', methods=['POST'])
def new_session():
    """Create a new chat session and make it active."""
    global active_session, response_times
    sid = create_session("New Chat")
    active_session = sid
    response_times = []
    return jsonify({'session_id': sid, 'title': 'New Chat'})


@app.route('/sessions/<sid>/load', methods=['POST'])
def load_session(sid):
    """Switch to an existing session."""
    global active_session, response_times
    if sid not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    active_session = sid
    response_times = []
    s = sessions[sid]
    return jsonify({
        'session_id': sid,
        'title':      s['title'],
        'history':    s['history']
    })


@app.route('/sessions/<sid>', methods=['DELETE'])
def delete_session(sid):
    """Delete a session."""
    global active_session
    if sid not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    del sessions[sid]
    if active_session == sid:
        active_session = None
    return jsonify({'message': 'Session deleted'})


@app.route('/sessions/<sid>/rename', methods=['POST'])
def rename_session(sid):
    """Rename a session."""
    if sid not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    data  = request.json
    title = data.get('title', '').strip()
    if not title:
        return jsonify({'error': 'Title cannot be empty'}), 400
    sessions[sid]['title'] = title
    return jsonify({'message': 'Renamed', 'title': title})


# ════════════════════════════════════════════════════════
#  ROUTES — CHAT
# ════════════════════════════════════════════════════════

@app.route('/chat', methods=['POST'])
def chat():
    global active_session
    try:
        data         = request.json
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Auto-create session if none active
        if not active_session or active_session not in sessions:
            active_session = create_session(auto_title(user_message))

        history = sessions[active_session]['history']

        # Auto-set title from first user message
        if len(history) == 0:
            sessions[active_session]['title'] = auto_title(user_message)

        history.append({
            'role':      'user',
            'content':   user_message,
            'timestamp': datetime.now().isoformat()
        })

        context = get_relevant_context(user_message) if vectorstore else None

        start_time  = time.time()
        bot_message = query_groq(user_message, history, context=context)
        elapsed     = round(time.time() - start_time, 2)

        is_offline = False
        if bot_message is None:
            bot_message = offline_response(user_message)
            elapsed     = 0.0
            is_offline  = True

        response_times.append(elapsed)
        source_tag = f"\n\n📄 *Answered using: {doc_name}*" if context and doc_name else ""

        history.append({
            'role':          'assistant',
            'content':       bot_message,
            'timestamp':     datetime.now().isoformat(),
            'response_time': elapsed
        })

        return jsonify({
            'response':      bot_message + source_tag,
            'response_time': elapsed,
            'is_offline':    is_offline,
            'used_document': context is not None,
            'session_id':    active_session,
            'session_title': sessions[active_session]['title'],
            'timestamp':     datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error in /chat: {e}")
        return jsonify({'error': 'An error occurred.'}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    if not response_times:
        return jsonify({'total_messages':0,'avg_time':0,'fastest':0,'slowest':0,'all_times':[]})
    return jsonify({
        'total_messages': len(response_times),
        'avg_time':  round(sum(response_times)/len(response_times), 2),
        'fastest':   round(min(response_times), 2),
        'slowest':   round(max(response_times), 2),
        'all_times': response_times[-20:]
    })


@app.route('/upload_doc', methods=['POST'])
def upload_doc():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        allowed = {'pdf', 'txt', 'md'}
        ext = file.filename.rsplit('.', 1)[-1].lower()
        if ext not in allowed:
            return jsonify({'error': 'Only PDF, TXT, MD supported'}), 400
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
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
    global vectorstore, doc_name
    vectorstore = None
    doc_name    = None
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
    return jsonify({'message': 'Document cleared'})


@app.route('/doc_status', methods=['GET'])
def doc_status():
    return jsonify({'loaded': vectorstore is not None, 'filename': doc_name})


@app.route('/clear', methods=['POST'])
def clear_history():
    global active_session, response_times
    if active_session and active_session in sessions:
        sessions[active_session]['history'] = []
    response_times = []
    return jsonify({'message': 'Conversation cleared'})


# ════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("🚀 Starting IntelliChat Server...")
    print("✅ Groq API Mode: Enabled")
    print("🧠 RAG Mode: LangChain + ChromaDB")
    print("⏱️  Response Time Tracking: Enabled")
    print("💬 Multi-Session Support: Enabled")
    print("🌐 Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)