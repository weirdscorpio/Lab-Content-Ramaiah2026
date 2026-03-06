#!/usr/bin/env python3
"""
RAG AI Assistant - Interactive RAG Chat Interface
"""

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import os
import sys
from datetime import datetime
import json
import time

# Add core modules to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vector_engine import VectorEngine
from core.chat_engine import ChatEngine
from core.document_processor import DocumentProcessor

# Configure Flask to use the correct template and static directories
template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'template')
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Initialize RAG components
print("\n" + "="*60)
print("\n[INIT] Loading RAG components...")

vector_engine = VectorEngine()
print("[INIT] Vector engine ready")

chat_engine = ChatEngine(vector_engine)
print("[INIT] Chat engine ready")

doc_processor = DocumentProcessor(vector_engine)
print("[INIT] Document processor ready")

@app.route('/', methods=['GET'])
def index():
    """Render the chat interface"""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get response from RAG system
        response = chat_engine.get_response(user_message)
        
        return jsonify({
            'response': response['answer'],
            'sources': response['sources'],
            'confidence': response['confidence'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Handle chat messages with streaming response"""
    def generate():
        try:
            data = request.json
            user_message = data.get('message', '')
            
            if not user_message:
                yield f"data: {json.dumps({'error': 'No message provided'})}\n\n"
                return
            
            # Send initial event
            yield f"data: {json.dumps({'event': 'start'})}\n\n"
            
            # Get response from RAG system
            response = chat_engine.get_response(user_message)
            
            # Stream the response word by word
            words = response['answer'].split()
            for i, word in enumerate(words):
                time.sleep(0.05)  # Small delay for streaming effect
                yield f"data: {json.dumps({'event': 'token', 'content': word + ' '})}\n\n"
            
            # Send sources at the end
            yield f"data: {json.dumps({'event': 'sources', 'sources': response['sources'], 'confidence': response['confidence']})}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'event': 'done'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/api/status', methods=['GET'])
def status():
    """Get system status"""
    try:
        stats = vector_engine.get_stats()
        return jsonify({
            'status': 'operational',
            'documents': stats['total_documents'],
            'chunks': stats['total_chunks'],
            'last_updated': stats['last_updated']
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def main():    # Initialize database with documents on first run
    if not vector_engine.is_initialized():
        print("First run detected. Processing all documents...")
        doc_processor.process_all_documents()
        print("Document processing complete!")
    
    # Run the app
    app.run(host='127.0.0.1', port=5252, debug=True)

main()