#!/usr/bin/env python3
"""
Flask Web UI for RAG Chatbot
Simple interface with Adani company colors (Navy Blue & Orange)
"""

import os
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
import uuid

load_dotenv()

from pdf_processor import extract_text_from_pdf, get_pdf_metadata
from chunker import create_chunks
from retriever import HybridRetriever
from chat_agent import ConversationalAgent

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global chatbot instance
chatbot = None


def get_chatbot():
    """Initialize or return existing chatbot instance."""
    global chatbot
    if chatbot is None:
        retriever = HybridRetriever(
            collection_name="document_chunks",
            persist_directory="./chroma_db"
        )
        chatbot = {
            'retriever': retriever,
            'agents': {}  # Per-session agents
        }
        # Try to load existing index
        try:
            collections = retriever.chroma_client.list_collections()
            for col in collections:
                if col.name == "document_chunks":
                    count = retriever.chroma_client.get_collection("document_chunks").count()
                    if count > 0:
                        retriever.collection = retriever.chroma_client.get_collection("document_chunks")
                        retriever._rebuild_bm25_from_collection()
                        chatbot['indexed'] = True
                        break
        except:
            chatbot['indexed'] = False
    return chatbot


def get_agent(session_id):
    """Get or create agent for session."""
    bot = get_chatbot()
    if session_id not in bot['agents']:
        bot['agents'][session_id] = ConversationalAgent(
            retriever=bot['retriever'],
            model="gpt-4o"  # Best model for natural conversation
        )
    return bot['agents'][session_id]


@app.route('/')
def index():
    """Render main chat interface."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    data = request.json
    message = data.get('message', '').strip()
    show_debug = data.get('show_debug', True)

    if not message:
        return jsonify({'error': 'Empty message'}), 400

    session_id = session.get('session_id', str(uuid.uuid4()))
    bot = get_chatbot()

    if not bot.get('indexed'):
        return jsonify({
            'error': 'No document indexed. Please upload a PDF first.',
            'answer': 'Please upload a PDF document first using the upload button.',
            'retrieved': []
        })

    try:
        agent = get_agent(session_id)
        answer, retrieved_chunks = agent.ask(message, top_k=5, show_debug=False)

        # Format retrieved chunks for display
        retrieved_display = []
        for chunk in retrieved_chunks:
            retrieved_display.append({
                'citation': chunk['citation'],
                'page': chunk['page_num'],
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'][:400] + '...' if len(chunk['text']) > 400 else chunk['text'],
                'vector_score': chunk.get('vector_score', 0),
                'bm25_score': chunk.get('bm25_score', 0),
                'combined_score': chunk.get('combined_score', 0)
            })

        return jsonify({
            'answer': answer,
            'retrieved': retrieved_display if show_debug else []
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload():
    """Handle multiple PDF uploads."""
    if 'files' not in request.files and 'file' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    # Support both single file (legacy) and multiple files
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        # Fallback to single file
        if 'file' in request.files:
            files = [request.files['file']]
        else:
            return jsonify({'error': 'No files selected'}), 400

    # Filter valid PDF files
    pdf_files = [f for f in files if f.filename and f.filename.lower().endswith('.pdf')]

    if not pdf_files:
        return jsonify({'error': 'No valid PDF files provided. Only PDF files are supported.'}), 400

    try:
        os.makedirs('./uploads', exist_ok=True)
        bot = get_chatbot()

        all_chunks = []
        total_pages = 0
        processed_files = []

        for file in pdf_files:
            # Save uploaded file
            upload_path = os.path.join('./uploads', file.filename)
            file.save(upload_path)

            # Extract text from PDF
            pages_data = extract_text_from_pdf(upload_path)

            # Add source filename to page data for better citations
            for page in pages_data:
                page['source_file'] = file.filename

            # Create chunks for this file
            file_chunks = create_chunks(pages_data, chunk_size=500, chunk_overlap=100)

            # Update chunk citations to include filename for multi-file scenarios
            if len(pdf_files) > 1:
                short_name = file.filename[:20] + '...' if len(file.filename) > 20 else file.filename
                for chunk in file_chunks:
                    chunk['source_file'] = file.filename
                    chunk['citation'] = f"[{short_name}:p{chunk['page_num']}:c{chunk['chunk_id']}]"

            all_chunks.extend(file_chunks)
            total_pages += len(pages_data)
            processed_files.append(file.filename)

        # Reindex chunk IDs to be unique across all files
        for i, chunk in enumerate(all_chunks):
            chunk['chunk_id'] = i

        # Index all chunks
        bot['retriever'].index_chunks(all_chunks, force_reindex=True)
        bot['indexed'] = True

        # Clear all agents (new documents)
        bot['agents'] = {}

        return jsonify({
            'success': True,
            'message': f'Indexed {len(all_chunks)} chunks from {total_pages} pages across {len(pdf_files)} file(s)',
            'files': processed_files,
            'file_count': len(pdf_files),
            'chunk_count': len(all_chunks),
            'page_count': total_pages
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/clear', methods=['POST'])
def clear_history():
    """Clear conversation history for current session."""
    session_id = session.get('session_id')
    if session_id:
        bot = get_chatbot()
        if session_id in bot['agents']:
            bot['agents'][session_id].clear_history()
    return jsonify({'success': True})


@app.route('/status')
def status():
    """Check system status."""
    bot = get_chatbot()
    indexed = bot.get('indexed', False)
    chunk_count = 0

    if indexed:
        try:
            chunk_count = bot['retriever'].collection.count()
        except:
            pass

    return jsonify({
        'indexed': indexed,
        'chunk_count': chunk_count
    })


# Create templates directory and HTML template
TEMPLATE_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Q&A - RAG Chatbot</title>
    <style>
        :root {
            --adani-navy: #003366;
            --adani-navy-dark: #002244;
            --adani-orange: #FF6600;
            --adani-orange-light: #FF8833;
            --adani-white: #FFFFFF;
            --adani-gray: #F5F5F5;
            --adani-gray-dark: #E0E0E0;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--adani-gray);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Header */
        .header {
            background: linear-gradient(135deg, var(--adani-navy) 0%, var(--adani-navy-dark) 100%);
            color: var(--adani-white);
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }

        .header h1 {
            font-size: 1.5rem;
            font-weight: 600;
        }

        .header-actions {
            display: flex;
            gap: 10px;
        }

        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s ease;
        }

        .btn-orange {
            background: var(--adani-orange);
            color: var(--adani-white);
        }

        .btn-orange:hover {
            background: var(--adani-orange-light);
        }

        .btn-outline {
            background: transparent;
            border: 2px solid var(--adani-white);
            color: var(--adani-white);
        }

        .btn-outline:hover {
            background: var(--adani-white);
            color: var(--adani-navy);
        }

        /* Main Container */
        .main-container {
            display: flex;
            flex: 1;
            overflow: hidden;
        }

        /* Chat Panel */
        .chat-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--adani-white);
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }

        .message {
            margin-bottom: 20px;
            max-width: 85%;
        }

        .message.user {
            margin-left: auto;
        }

        .message-content {
            padding: 12px 16px;
            border-radius: 12px;
            line-height: 1.5;
        }

        .message.user .message-content {
            background: var(--adani-navy);
            color: var(--adani-white);
            border-bottom-right-radius: 4px;
        }

        .message.assistant .message-content {
            background: var(--adani-gray);
            color: #333;
            border-bottom-left-radius: 4px;
        }

        .message-label {
            font-size: 0.75rem;
            color: #888;
            margin-bottom: 4px;
        }

        .message.user .message-label {
            text-align: right;
        }

        /* Chat Input */
        .chat-input-container {
            padding: 15px 20px;
            background: var(--adani-white);
            border-top: 1px solid var(--adani-gray-dark);
        }

        .chat-input-wrapper {
            display: flex;
            gap: 10px;
        }

        .chat-input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid var(--adani-gray-dark);
            border-radius: 25px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.3s;
        }

        .chat-input:focus {
            border-color: var(--adani-orange);
        }

        .send-btn {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: var(--adani-orange);
            color: var(--adani-white);
            border: none;
            cursor: pointer;
            font-size: 1.2rem;
            transition: background 0.3s;
        }

        .send-btn:hover {
            background: var(--adani-orange-light);
        }

        .send-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }

        /* Debug Panel */
        .debug-panel {
            width: 400px;
            background: var(--adani-navy-dark);
            color: var(--adani-white);
            overflow-y: auto;
            transition: width 0.3s;
        }

        .debug-panel.hidden {
            width: 0;
            padding: 0;
        }

        .debug-header {
            padding: 15px;
            background: var(--adani-navy);
            font-weight: 600;
            position: sticky;
            top: 0;
        }

        .debug-content {
            padding: 15px;
        }

        .debug-chunk {
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
        }

        .debug-chunk-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 0.85rem;
        }

        .debug-citation {
            background: var(--adani-orange);
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
        }

        .debug-scores {
            color: #aaa;
            font-size: 0.75rem;
        }

        .debug-text {
            font-size: 0.85rem;
            line-height: 1.4;
            color: #ddd;
        }

        /* Upload Modal */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }

        .modal.active {
            display: flex;
        }

        .modal-content {
            background: var(--adani-white);
            padding: 30px;
            border-radius: 12px;
            width: 90%;
            max-width: 500px;
        }

        .modal-header {
            color: var(--adani-navy);
            margin-bottom: 20px;
            font-size: 1.3rem;
        }

        .upload-area {
            border: 2px dashed var(--adani-gray-dark);
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.3s;
        }

        .upload-area:hover {
            border-color: var(--adani-orange);
        }

        .upload-area.dragover {
            border-color: var(--adani-orange);
            background: rgba(255,102,0,0.05);
        }

        .upload-icon {
            font-size: 3rem;
            color: var(--adani-orange);
            margin-bottom: 10px;
        }

        .upload-text {
            color: #666;
        }

        .upload-progress {
            display: none;
            margin-top: 20px;
        }

        .progress-bar {
            height: 8px;
            background: var(--adani-gray);
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: var(--adani-orange);
            width: 0%;
            transition: width 0.3s;
        }

        .progress-text {
            margin-top: 10px;
            text-align: center;
            color: #666;
        }

        /* Status indicator */
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }

        .status-dot.ready {
            background: #4CAF50;
        }

        .status-dot.not-ready {
            background: #F44336;
        }

        /* Toggle button */
        .toggle-debug {
            background: var(--adani-navy);
            color: var(--adani-white);
            border: none;
            padding: 8px 12px;
            cursor: pointer;
            position: fixed;
            right: 0;
            top: 50%;
            transform: translateY(-50%);
            border-radius: 8px 0 0 8px;
            z-index: 100;
        }

        /* Loading animation */
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: var(--adani-orange);
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Markdown-like formatting in answers */
        .message-content strong {
            color: var(--adani-navy);
        }

        .message.assistant .message-content strong {
            color: var(--adani-navy);
        }

        /* Welcome message */
        .welcome {
            text-align: center;
            padding: 40px;
            color: #888;
        }

        .welcome h2 {
            color: var(--adani-navy);
            margin-bottom: 10px;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }

        ::-webkit-scrollbar-thumb {
            background: #ccc;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #aaa;
        }
    </style>
</head>
<body>
    <header class="header">
        <h1>📄 Document Q&A Assistant</h1>
        <div class="header-actions">
            <span id="status"><span class="status-dot not-ready"></span>No document</span>
            <button class="btn btn-orange" onclick="openUploadModal()">📤 Upload PDFs</button>
            <button class="btn btn-outline" onclick="clearChat()">🗑️ Clear Chat</button>
            <button class="btn btn-outline" onclick="toggleDebug()">🔍 Debug</button>
        </div>
    </header>

    <div class="main-container">
        <div class="chat-panel">
            <div class="chat-messages" id="chatMessages">
                <div class="welcome">
                    <h2>Welcome!</h2>
                    <p>Upload one or multiple PDF documents and start asking questions.</p>
                    <p>I'll provide answers with citations from the documents.</p>
                </div>
            </div>
            <div class="chat-input-container">
                <div class="chat-input-wrapper">
                    <input type="text" class="chat-input" id="chatInput"
                           placeholder="Ask a question about the document..."
                           onkeypress="handleKeyPress(event)">
                    <button class="send-btn" id="sendBtn" onclick="sendMessage()">➤</button>
                </div>
            </div>
        </div>

        <div class="debug-panel" id="debugPanel">
            <div class="debug-header">🔍 Retrieved Context</div>
            <div class="debug-content" id="debugContent">
                <p style="color: #aaa;">Retrieved chunks will appear here...</p>
            </div>
        </div>
    </div>

    <!-- Upload Modal -->
    <div class="modal" id="uploadModal">
        <div class="modal-content">
            <h2 class="modal-header">Upload PDF Documents</h2>
            <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📁</div>
                <div class="upload-text">
                    <strong>Click to upload</strong> or drag and drop<br>
                    <small>Select one or multiple PDF files</small>
                </div>
            </div>
            <input type="file" id="fileInput" accept=".pdf" multiple style="display: none" onchange="handleFileSelect(event)">
            <div id="selectedFiles" style="margin-top: 15px; display: none;">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">Selected files:</div>
                <div id="fileList" style="max-height: 150px; overflow-y: auto;"></div>
            </div>
            <div class="upload-progress" id="uploadProgress">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="progress-text" id="progressText">Processing...</div>
            </div>
            <div style="margin-top: 20px; text-align: right;">
                <button class="btn btn-outline" style="color: #333; border-color: #ccc;" onclick="closeUploadModal()">Cancel</button>
            </div>
        </div>
    </div>

    <script>
        let debugVisible = true;

        // Check status on load
        document.addEventListener('DOMContentLoaded', checkStatus);

        function checkStatus() {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    const statusEl = document.getElementById('status');
                    if (data.indexed) {
                        statusEl.innerHTML = `<span class="status-dot ready"></span>${data.chunk_count} chunks indexed`;
                    } else {
                        statusEl.innerHTML = `<span class="status-dot not-ready"></span>No document`;
                    }
                });
        }

        function toggleDebug() {
            const panel = document.getElementById('debugPanel');
            debugVisible = !debugVisible;
            panel.classList.toggle('hidden');
        }

        function openUploadModal() {
            document.getElementById('uploadModal').classList.add('active');
        }

        function closeUploadModal() {
            document.getElementById('uploadModal').classList.remove('active');
            document.getElementById('uploadProgress').style.display = 'none';
        }

        // Drag and drop
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                uploadFiles(files);
            }
        });

        function handleFileSelect(event) {
            const files = event.target.files;
            if (files.length > 0) {
                uploadFiles(files);
            }
        }

        function uploadFiles(files) {
            // Filter PDF files
            const pdfFiles = Array.from(files).filter(f => f.name.toLowerCase().endsWith('.pdf'));

            if (pdfFiles.length === 0) {
                alert('Please upload PDF files only');
                return;
            }

            // Show selected files
            const selectedFilesDiv = document.getElementById('selectedFiles');
            const fileListDiv = document.getElementById('fileList');
            selectedFilesDiv.style.display = 'block';
            fileListDiv.innerHTML = pdfFiles.map(f =>
                `<div style="padding: 5px 10px; background: #f0f0f0; border-radius: 4px; margin-bottom: 5px; font-size: 0.85rem;">
                    📄 ${f.name} <span style="color: #888;">(${(f.size / 1024).toFixed(1)} KB)</span>
                </div>`
            ).join('');

            // Create form data with multiple files
            const formData = new FormData();
            pdfFiles.forEach(file => {
                formData.append('files', file);
            });

            document.getElementById('uploadProgress').style.display = 'block';
            document.getElementById('progressFill').style.width = '20%';
            document.getElementById('progressText').textContent = `Uploading ${pdfFiles.length} file(s)...`;

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('progressFill').style.width = '100%';
                    document.getElementById('progressText').textContent = data.message;
                    setTimeout(() => {
                        closeUploadModal();
                        checkStatus();
                        clearChat();
                        // Reset file input and selected files display
                        document.getElementById('fileInput').value = '';
                        document.getElementById('selectedFiles').style.display = 'none';
                    }, 2000);
                } else {
                    document.getElementById('progressFill').style.width = '0%';
                    document.getElementById('progressText').textContent = 'Error: ' + data.error;
                }
            })
            .catch(err => {
                document.getElementById('progressFill').style.width = '0%';
                document.getElementById('progressText').textContent = 'Upload failed: ' + err;
            });
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message) return;

            // Add user message
            addMessage(message, 'user');
            input.value = '';

            // Disable input while processing
            const sendBtn = document.getElementById('sendBtn');
            sendBtn.disabled = true;
            sendBtn.innerHTML = '<span class="loading"></span>';

            // Send to server
            fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message, show_debug: debugVisible})
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    addMessage(data.error, 'assistant');
                } else {
                    addMessage(data.answer, 'assistant');
                    updateDebugPanel(data.retrieved);
                }
            })
            .catch(err => {
                addMessage('Error: ' + err, 'assistant');
            })
            .finally(() => {
                sendBtn.disabled = false;
                sendBtn.innerHTML = '➤';
            });
        }

        function addMessage(content, type) {
            const container = document.getElementById('chatMessages');

            // Remove welcome message if present
            const welcome = container.querySelector('.welcome');
            if (welcome) welcome.remove();

            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + type;

            const label = type === 'user' ? 'You' : 'Assistant';

            // Format content (basic markdown-like)
            let formattedContent = content
                .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\n/g, '<br>');

            messageDiv.innerHTML = `
                <div class="message-label">${label}</div>
                <div class="message-content">${formattedContent}</div>
            `;

            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }

        function updateDebugPanel(chunks) {
            const content = document.getElementById('debugContent');

            if (!chunks || chunks.length === 0) {
                content.innerHTML = '<p style="color: #aaa;">No chunks retrieved</p>';
                return;
            }

            let html = '';
            chunks.forEach((chunk, i) => {
                html += `
                    <div class="debug-chunk">
                        <div class="debug-chunk-header">
                            <span class="debug-citation">${chunk.citation}</span>
                            <span class="debug-scores">
                                V: ${chunk.vector_score.toFixed(3)} |
                                B: ${chunk.bm25_score.toFixed(3)} |
                                C: ${chunk.combined_score.toFixed(4)}
                            </span>
                        </div>
                        <div class="debug-text">${chunk.text}</div>
                    </div>
                `;
            });

            content.innerHTML = html;
        }

        function clearChat() {
            fetch('/clear', {method: 'POST'})
                .then(() => {
                    const container = document.getElementById('chatMessages');
                    container.innerHTML = `
                        <div class="welcome">
                            <h2>Chat cleared!</h2>
                            <p>Ask a new question about the document.</p>
                        </div>
                    `;
                    document.getElementById('debugContent').innerHTML =
                        '<p style="color: #aaa;">Retrieved chunks will appear here...</p>';
                });
        }
    </script>
</body>
</html>
'''


def create_templates():
    """Create templates directory and write HTML."""
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w') as f:
        f.write(TEMPLATE_HTML)


if __name__ == '__main__':
    # Create templates on startup
    create_templates()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        exit(1)

    print("\n" + "="*50)
    print("RAG Chatbot Web UI")
    print("="*50)
    print("\nOpen http://localhost:5001 in your browser")
    print("Press Ctrl+C to stop\n")

    app.run(host="0.0.0.0", port=5001, debug=True)

