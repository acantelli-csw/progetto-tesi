## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Cantellos/adaptive-rag-pipeline.git
cd adaptive-rag-pipeline

# With uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env and fill in your API key, DB credentials, etc.
```

### 3. Configure your domain

Open `config.yaml` and set the values for your knowledge base:

```yaml
domain:
  name: "My Knowledge Base"          # Name of the system/domain
  document_type: "Document"          # What individual documents are called
  document_url_template: ""          # Optional link template: "https://my-intranet/docs/{id}"
  language: "english"                # Language for BM25 stemming

input:
  mode: "folder"                     # "folder" or "database"

folder_input:
  path: "./documents"                # Put your .pdf and .docx files here
```

### 4. Ingest documents

Place your `.pdf` and `.docx` files in the `documents/` folder (or configure a database source), then run:

```bash
python main/file_embedding/files_processing.py
```

This will:
1. Extract text (+ OCR for embedded images) from each file
2. Split each document into chunks
3. Compute and store embeddings in SQL Server
4. Build the BM25 reverse index

### 5. Launch the chat interface

```bash
streamlit run main/llm/app.py
```

Open your browser at `http://localhost:8501`, register an account and start querying your knowledge base.