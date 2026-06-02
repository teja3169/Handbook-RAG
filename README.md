# Minimal PDF RAG

A small ready-to-run RAG app that:
- uploads PDF files
- extracts text page by page
- chunks and embeds the text
- stores embeddings in a local JSON vector store
- answers questions in a browser chat UI

## Project structure

```text
rag-minimal/
├── main.py
├── ingest.py
├── rag.py
├── requirements.txt
├── .env.example
├── data/
├── vector_store/
└── templates/
    └── index.html
```

## Setup

1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Create your env file

```bash
cp .env.example .env
```

4. Add your OpenAI API key in `.env`

## Run

1. Start the web app

```bash
uvicorn main:app --reload
```

2. Open the browser at:

```text
http://127.0.0.1:8000
```

3. Upload one or more PDFs from the UI

4. Index PDFs

```bash
python ingest.py
```

5. Ask questions in the chat box
