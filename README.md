# RAG App on Endee

This repository is centered around a Retrieval-Augmented Generation (RAG) application in `rag_app/`.

The app lets you:

- Upload a PDF or TXT document
- Chunk and embed the content
- Store embeddings in Endee vector database
- Retrieve relevant context with semantic search
- Generate answers with Groq LLM

## What Is In This Repo

- `rag_app/`: Your Streamlit RAG application (main project)
- `src/`, `docs/`, `infra/`, `third_party/`: Endee source code and infrastructure

In short: this is your RAG project with Endee embedded as the vector database backend.

## Architecture

1. User uploads document text (PDF/TXT) in Streamlit UI.
2. Text is chunked and converted to embeddings with `sentence-transformers` (`all-MiniLM-L6-v2`).
3. Embeddings are stored in an Endee index.
4. User asks a question.
5. Top relevant chunks are retrieved from Endee.
6. Retrieved context is sent to Groq for grounded answer generation.

## Project Focus: `rag_app/`

Main files:

- `rag_app/main.py`: Streamlit app, indexing flow, retrieval, and answer generation
- `rag_app/tools.py`: PDF/TXT parsing and chunking utilities
- `rag_app/requirements.txt`: Python dependencies

## Prerequisites

- Python 3.10+
- A running Endee server (default expected at `http://localhost:8080`)
- Groq API key

## Run Endee Server

Use one of the official Endee startup paths from this repository docs:

- `docs/getting-started.md`
- Hosted docs: <https://docs.endee.io/quick-start>

Quick Docker example:

```bash
docker run \
  --ulimit nofile=100000:100000 \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  --restart unless-stopped \
  endeeio/endee-server:latest
```

## Run The RAG App

From the repository root:

```bash
cd rag_app
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file inside `rag_app/`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Start the app:

```bash
streamlit run main.py
```

Then open the local Streamlit URL shown in terminal (usually `http://localhost:8501`).

## Usage Flow

1. Upload a PDF/TXT file or paste text.
2. Click `Process & Create Index`.
3. Ask questions in the chat section.
4. Expand `Retrieved Context` to inspect supporting chunks.

## Endee Note

Endee is an open-source vector database used by this app for indexing and nearest-neighbor retrieval.

- Endee GitHub: <https://github.com/endee-io/endee>
- Endee Docs: <https://docs.endee.io/quick-start>
- Endee Website: <https://endee.io/>

## License

This repository includes Endee, which is licensed under Apache License 2.0. See `LICENSE` for details.
