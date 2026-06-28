# AskDoc

AI document reviewer built with Django, FAISS, and a local LLM.

## Overview

AskDoc lets users upload documents and ask questions about each document using a chat-like interface. Each uploaded file becomes its own session, with a separate document index so answers stay tied to the selected document.

## Features

- Upload PDF, TXT, CSV, and HTML documents
- Document ingestion with chunking and embeddings
- Per-document FAISS indexing and retrieval
- Recent document sessions in a sidebar
- Chat interface with session-specific context
- Local LLM generation for answer synthesis
- OCR support for scanned PDFs

## Architecture & Flow

1. Upload document
2. Generate text from file using `pypdf`, `pdf2image`, `pytesseract`, or `BeautifulSoup`
3. Chunk text and create embeddings with `sentence-transformers`
4. Store document embeddings in a FAISS index
5. Select a recent document session from the sidebar
6. Send question + `session_id` to backend
7. Retrieve document-specific context from FAISS
8. Generate answer with a local LLM

## Tech Stack

- Python
- Django
- FAISS
- SentenceTransformers
- transformers
- PyTorch
- pypdf
- pdf2image
- pytesseract
- BeautifulSoup

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python manage.py runserver
```

## Usage

1. Open `http://127.0.0.1:8000`
2. Click `+ New Document`
3. Upload a supported file
4. Select a recent document session
5. Ask questions in the composer
