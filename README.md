# RAG PDF Conversational Agent

This project is a Retrieval-Augmented Generation (RAG) based conversational agent that answers questions grounded strictly in the content of a provided PDF document.

The system retrieves relevant context from the document and generates answers only from that retrieved text. If the answer is not present in the document, it responds with **"Not found in the document."**

The solution works for earnings decks, reports, policies, and contracts.

---

## Objective

Build a working conversational agent that:

- Ingests a local PDF
- Indexes the document using retrieval (vector-based)
- Supports multi-turn conversational Q&A
- Produces grounded answers with citations (page numbers)
- Prevents hallucinations

---

## Features

### A) Ingestion & Indexing
- Input: local PDF file (e.g. `./doc.pdf`)
- Extracts text page-by-page
- Chunks text and stores metadata (page number, chunk id)
- Builds a vector-based retrieval index (Chroma)

### B) Conversational Q&A
- Multi-turn chat loop
- Conversation history is preserved
- Each answer is grounded only in retrieved document context

### C) Grounded Answers with Citations
Each response includes:
- A short answer
- Citations pointing to the source document  
  Example: `[p13]` or `[p13:c42]`

If the answer is not found:
Not found in the document.

### D) Retrieval Visibility (Debug)
For every question, the system shows:
- Top-k retrieved chunks
- Page and chunk IDs
- Retrieval scores (if available)

---

## Embedding & Storage Behavior

- Document embeddings are stored locally in the `chroma_db/` directory.
- If embeddings already exist, they are **reused** to avoid re-processing the PDF.
- If the PDF changes or the index is deleted, embeddings are rebuilt automatically.
- The `chroma_db/` folder is **not committed** to GitHub.

---

## Project Structure
```bash
rag-pdf-chatbot/
├── app.py                 # Web UI (Flask)
├── main.py                # CLI entry point
├── chat_agent.py          # Conversational logic (multi-turn, grounding)
├── pdf_processor.py       # PDF text extraction (page-wise)
├── chunker.py             # Text chunking + metadata (page, chunk id)
├── retriever.py           # Retrieval logic (Chroma vector search)
├── download_sample.py     # Download sample PDF for testing
├── test_acceptance.py     # Acceptance tests
├── templates/             # HTML templates (if web UI is used)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```
---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt

2. Download sample PDF
python download_sample.py

3. Run the chatbot
python main.py


or (web UI)

python app.py
```
---

Example Questions

What are the major business segments discussed?

What is the consolidated total income in H1-26?

What drivers are mentioned for EBITDA changes?

What is the CEO’s email address?
