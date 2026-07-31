# Mental Health Chatbot

An AI-powered mental health chatbot pipeline that performs Dialog Analysis (emotion, severity, intent, cause/effect) and Knowledge Graph Retrieval (RAG from 5 psychology books) to generate empathetic responses via a local Gemma LLM (Ollama).

## 🚀 Setup Instructions for Teammates

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed on your machine

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate Knowledge Graph Data (Important!)
Because the generated knowledge graph and vector embeddings are too large for GitHub (>600MB), they are intentionally not tracked in version control. **You must generate them locally on your machine.** 

Ensure you have the source PDFs in `data/books/`, then run these two commands **once**:
```bash
# Step A: Parses books and builds the word co-occurrence graph (~1-2 minutes)
python -m knowledge.ingest

# Step B: Generates dense vector embeddings for semantic search
python -m knowledge.ingest_embeddings
```

### 4. Pull the Local LLM
We use Google's `gemma3:4b` model locally via Ollama to generate responses, meaning no API keys are required. Make sure the Ollama app is running, then pull the model:
```bash
ollama pull gemma3:4b
```
*(Note: This is a ~3.3GB download and may take a few minutes depending on your internet connection)*

### 5. Run the Chatbot
Once the data is generated and the model is pulled, you can start the end-to-end chatbot pipeline:
```bash
python run_with_summarizer.py
```