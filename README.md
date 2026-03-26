# Research Assistant RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI, LangChain, and ChromaDB. This service allows you to query local PDF documents, utilizing a two-stage retrieval pipeline (Dense Vector Search + Cross-Encoder Reranker) for highly precise context retrieval.

## Features
- **Stateless API:** Engineered on FastAPI with strict Pydantic schemas.
- **Advanced Retrieval:** Employs HuggingFace Embeddings paired with a Cross-Encoder Reranker to filter and surface the absolute best context chunks.
- **Local Vector DB:** Uses Chroma for fast, persistent document similarity searches.
- **LLM Caching:** Implements `InMemoryCache` to prevent redundant API calls to OpenAI, saving both costs and time.
- **Dedicated Ingestion:** An automated script for parsing, splitting, and vectorizing PDFs.

---

## Setup & Local Installation

### 1. Install Dependencies
Ensure you have Python installed. Inside the root directory, install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a file named `.env` in the root repository and add your OpenAI API key for `gpt-4o-mini`:

```env
OPENAI_API_KEY="sk-your-openai-api-key-here"
```

### 3. Add Your Data
Place any research `.pdf` files you want the system to be able to read inside the `data/papers/` directory.

---

## How to Run

### Step 1: Ingest the Documents (Do this whenever you add new PDFs)
Before querying, you must process the raw PDFs into the searchable vector database. Open your terminal and run:

```bash
python -m app.ingestion
```

### Step 2: Start the Server
Launch the FastAPI web server by double clicking the batch file or running:

```bash
run.bat
```

*(Alternatively, run `uvicorn app.main:app --host 0.0.0.0 --port 8000`)*

---

## API Usage

Once the server is running, you can explore and test the endpoints visually by visiting the automatically generated Swagger UI page at: 
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

### Querying the AI

**Endpoint:**
`POST /query`

**Payload Format:**
```json
{
  "question": "What is the primary methodology used in the latest paper?"
}
```

**Successful Response Example:**
The response strictly returns the answer from the LLM alongside the exact source pages the bot referenced.
```json
{
  "answer": "The primary methodology used relies on a two-stage retriever with cross-encoders...",
  "sources": [
    {
      "source": "data/papers/study.pdf",
      "page": 4
    }
  ]
}
```
