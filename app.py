from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Option 1: Groq API (Fast and Free) - Get key from https://console.groq.com
GROQ_API_KEY = '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'
USE_GROQ = True  # Set to True to use Groq, False for offline mode

# Store conversation history
conversation_history = []

def query_groq(message):
    """Query Groq API"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Build conversation context - only include user and assistant messages
    messages = [{"role": "system", "content": "You are a helpful AI assistant. Be friendly and concise."}]
    
    # Add recent conversation history (exclude current message as it will be added)
    recent_history = [msg for msg in conversation_history[-6:] if msg['role'] in ['user', 'assistant']]
    for msg in recent_history:
        messages.append({"role": msg['role'], "content": msg['content']})
    
    # Add current user message
    messages.append({"role": "user", "content": message})
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            print(f"Groq API Error: {response.status_code} - {response.text}")
            return None
        data = response.json()
        return data['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Groq API Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return None

def offline_response(message):
    """Simple offline AI-like responses"""
    message_lower = message.lower()
    
    # Greetings
    if any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
        return "Hello! How can I help you today?"
    
    # How are you
    if any(phrase in message_lower for phrase in ['how are you', 'how do you do', 'whats up']):
        return "I'm doing great, thank you for asking! How can I assist you?"
    
    # Name questions
    if 'your name' in message_lower or 'who are you' in message_lower:
        return "I'm an AI chatbot assistant created for this college project. I'm here to help answer your questions!"
    
    # Help questions
    if any(word in message_lower for word in ['help', 'assist', 'support']):
        return "I'm here to help! You can ask me questions, have a conversation, or just chat. What would you like to know?"
    
    # Time questions
    if 'time' in message_lower:
        current_time = datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}."
    
    # Date questions
    if 'date' in message_lower or 'today' in message_lower:
        current_date = datetime.now().strftime("%B %d, %Y")
        return f"Today is {current_date}."
    
    # Thank you
    if any(word in message_lower for word in ['thank', 'thanks']):
        return "You're welcome! Is there anything else I can help you with?"
    
    # Goodbye
    if any(word in message_lower for word in ['bye', 'goodbye', 'see you']):
        return "Goodbye! Have a great day!"
    
    # Weather
    if 'weather' in message_lower:
        return "I don't have access to real-time weather data, but you can check your local weather forecast online!"
    
    # Default responses based on question marks
    if '?' in message:
        return "That's an interesting question! While I'm running in offline mode, I can help with basic queries. For more advanced responses, you can connect an AI API."
    
    # Generic responses
    responses = [
        "That's interesting! Tell me more about that.",
        "I understand what you're saying. Could you elaborate?",
        "Thanks for sharing that with me!",
        "I see. What else would you like to discuss?",
        "Interesting perspective! What makes you say that?",
    ]
    
    import random
    return random.choice(responses)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Add user message to history
        conversation_history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Get bot response
        if USE_GROQ and GROQ_API_KEY != 'your-groq-api-key-here':
            bot_message = query_groq(user_message)
            if bot_message is None:
                # Fallback to offline mode if API fails
                bot_message = offline_response(user_message)
                bot_message += " (API unavailable - using offline mode)"
        else:
            bot_message = offline_response(user_message)
        
        # Add bot response to history
        conversation_history.append({
            'role': 'assistant',
            'content': bot_message,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'response': bot_message,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'An error occurred. Please try again.'}), 500

@app.route('/clear', methods=['POST'])
def clear_history():
    global conversation_history
    conversation_history = []
    return jsonify({'message': 'Conversation cleared'})

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({'history': conversation_history})

if __name__ == '__main__':
    print("🤖 Starting AI Chatbot Server...")
    if USE_GROQ and GROQ_API_KEY != 'your-groq-api-key-here':
        print("✅ Groq API Mode: Enabled")
    else:
        print("📴 Offline Mode: Enabled (Rule-based responses)")
    print("🌐 Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)