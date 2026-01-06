# CogniDoc

**Intelligent Document Assistant** powered by Hybrid RAG (Vector + GraphRAG).

A document processing and retrieval pipeline that combines **Vector RAG** and **GraphRAG** for intelligent document querying. Converts PDFs into a searchable knowledge base with a professional chat interface.

## Features

- **Hybrid RAG**: Combines vector similarity search with knowledge graph traversal
- **GraphRAG**: Automatic entity/relationship extraction with community detection
- **Multi-Format Support**: PDF, PPTX, DOCX, XLSX, HTML, Markdown, images
- **Flexible Providers**: Mix and match LLM and embedding providers independently
- **YOLO Object Detection**: Automatically detects tables, pictures, text regions (optional)
- **Semantic Chunking**: Embedding-based coherent text chunks
- **Intelligent Query Routing**: LLM-based classification with smart skip logic
- **Clickable PDF References**: Response references link directly to source PDFs
- **No LangChain/LlamaIndex**: Direct Qdrant and provider integration

## Installation

```bash
# Basic installation (from GitHub)
pip install git+https://github.com/arielibaba/cognidoc.git

# With Gradio UI
pip install "cognidoc[ui] @ git+https://github.com/arielibaba/cognidoc.git"

# With YOLO detection
pip install "cognidoc[yolo] @ git+https://github.com/arielibaba/cognidoc.git"

# With local Ollama support
pip install "cognidoc[ollama] @ git+https://github.com/arielibaba/cognidoc.git"

# Full installation (all features)
pip install "cognidoc[all] @ git+https://github.com/arielibaba/cognidoc.git"
```

### Development Installation

```bash
git clone https://github.com/arielibaba/cognidoc.git
cd cognidoc
make install  # Uses uv package manager
# or
pip install -e ".[all,dev]"
```

## Quick Start

### Python API

```python
from cognidoc import CogniDoc

# Simple usage (Gemini LLM + Ollama embeddings)
doc = CogniDoc()
doc.ingest("./documents/")
result = doc.query("What is the main topic?")
print(result.answer)

# Full cloud mode (no local dependencies)
doc = CogniDoc(
    llm_provider="openai",
    embedding_provider="openai",
)

# Hybrid mode (mix providers)
doc = CogniDoc(
    llm_provider="gemini",
    embedding_provider="ollama",
    use_yolo=False,  # Skip YOLO, use simple extraction
)

# Launch web interface
doc.launch_ui(port=7860, share=True)
```

### CLI

```bash
# Initialize project (copy schema/prompts templates)
cognidoc init --schema --prompts

# Ingest documents
cognidoc ingest ./documents --llm gemini --embedding ollama

# Cloud-only mode
cognidoc ingest ./documents --llm openai --embedding openai

# Without YOLO (simpler extraction)
cognidoc ingest ./documents --no-yolo

# Query
cognidoc query "Summarize the key findings"

# Launch web UI
cognidoc serve --port 7860 --share
```

### Interactive Setup Wizard

For guided configuration:

```bash
python -m src.setup
```

## Provider Configuration

CogniDoc supports flexible provider mixing - use different providers for LLM and embeddings:

| Provider | LLM | Embeddings | Requires |
|----------|-----|------------|----------|
| **Gemini** | `gemini-2.0-flash` | `text-embedding-004` | `GEMINI_API_KEY` |
| **OpenAI** | `gpt-4o-mini` | `text-embedding-3-small` | `OPENAI_API_KEY` |
| **Anthropic** | `claude-3-haiku` | - | `ANTHROPIC_API_KEY` |
| **Ollama** | `granite3.3:8b` | `qwen3-embedding:0.6b` | Local Ollama server |

### Example Configurations

```python
# Full local (free, requires Ollama)
CogniDoc(llm_provider="ollama", embedding_provider="ollama")

# Full cloud (no local deps, API costs)
CogniDoc(llm_provider="gemini", embedding_provider="openai")

# Hybrid (best of both)
CogniDoc(llm_provider="gemini", embedding_provider="ollama")
```

## Architecture

### Ingestion Pipeline

```
Documents → PDF Conversion → Images (600 DPI) → YOLO Detection*
                                                      ↓
                                    Text/Table/Image Extraction
                                                      ↓
                                            Semantic Chunking
                                       (Parent + Child hierarchy)
                                                      ↓
                        ┌─────────────────────────────┴─────────────────────────────┐
                        ↓                                                           ↓
               Vector Embeddings                                        Entity/Relationship
               (Qdrant + BM25)                                              Extraction
                        ↓                                                           ↓
                        └─────────────────────────────┬─────────────────────────────┘
                                                      ↓
                                            Hybrid Retriever
```

*YOLO is optional - falls back to simple page-level extraction if not installed.

### Query Routing

| Query Type | Example | Vector | Graph |
|------------|---------|--------|-------|
| **FACTUAL** | "What is X?" | 70% | 30% |
| **RELATIONAL** | "Relationship between A and B?" | 20% | 80% |
| **EXPLORATORY** | "List all main topics" | 0% | 100% |
| **PROCEDURAL** | "How to configure?" | 80% | 20% |

## Configuration

### Environment Variables

```bash
# Provider selection
COGNIDOC_LLM_PROVIDER=gemini
COGNIDOC_EMBEDDING_PROVIDER=ollama
COGNIDOC_DATA_DIR=./data

# API Keys
GEMINI_API_KEY=your-key
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key

# Ollama (if using local)
OLLAMA_HOST=http://localhost:11434
```

### GraphRAG Schema

Customize entity extraction in `config/graph_schema.yaml`:

```yaml
domain:
  name: "your-domain"
  description: "Domain context for LLM extraction"

entity_types:
  - name: "Concept"
    description: "Abstract ideas"
    examples: ["machine learning", "ethics"]

relationship_types:
  - name: "RELATED_TO"
    description: "General relationship"
```

### Prompts

All LLM prompts are in `prompts/` directory and can be customized.

## Requirements

### Minimal (Cloud-only)

- Python 3.10+
- API key for at least one provider (Gemini, OpenAI, or Anthropic)

### Full Features

- [Ollama](https://ollama.ai/) for local inference (optional)
- [LibreOffice](https://www.libreoffice.org/) for Office document conversion
- YOLO model for advanced document detection (optional)

### Ollama Models (if using local)

```bash
ollama pull granite3.3:8b          # LLM
ollama pull qwen3-embedding:0.6b   # Embeddings
ollama pull qwen3-vl:8b-instruct   # Vision (optional)
```

## Project Structure

```
cognidoc/
├── src/cognidoc/           # Main package
│   ├── api.py              # CogniDoc class
│   ├── cli.py              # Command-line interface
│   ├── app.py              # Gradio interface
│   ├── pipeline/           # Ingestion pipeline
│   ├── retrieval/          # Hybrid retriever
│   └── providers/          # LLM/Embedding providers
├── config/
│   └── graph_schema.yaml   # GraphRAG configuration
├── prompts/                # Customizable prompts
└── data/                   # Document storage
    ├── pdfs/               # Input documents
    ├── indexes/            # Vector/graph indexes
    └── vector_store/       # Qdrant database
```

## Development

```bash
make format    # Format with black
make lint      # Run pylint
make refactor  # Format + lint
make test      # Run tests
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `qdrant-client` | Vector database |
| `networkx` | Knowledge graph |
| `gradio` | Web interface (optional) |
| `ultralytics` | YOLO detection (optional) |
| `ollama` | Local inference (optional) |
| `google-generativeai` | Gemini API |
| `openai` | OpenAI API |
| `anthropic` | Claude API |

## License

MIT
