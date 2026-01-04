# Advanced Hybrid RAG

A document processing and retrieval pipeline that converts PDFs into a searchable knowledge base with a chat interface.

## Features

- **Multi-modal Document Processing**: Handles text, tables, and images from PDFs
- **YOLO Object Detection**: Automatically detects and extracts tables, pictures, and text regions
- **Semantic Chunking**: Uses embeddings to create semantically coherent text chunks
- **Hierarchical Retrieval**: Parent-child document structure for context-aware search
- **LLM Reranking**: Improves retrieval quality using LLM-based relevance scoring
- **Local Inference**: All models run locally via Ollama

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai/) running locally
- Required Ollama models (run these commands):

```bash
ollama pull granite3.3:8b          # LLM for generation and reranking
ollama pull qwen3-vl:8b-instruct   # Vision model for image descriptions
ollama pull qwen3-embedding:0.6b   # Embedding model
ollama pull ibm/granite-docling:258m-bf16  # Document parsing
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/advanced-hybrid-rag.git
cd advanced-hybrid-rag

# Install dependencies (requires uv package manager)
make install

# Or using pip
pip install -e .
```

## Project Structure

```
advanced-hybrid-rag/
├── data/                      # All pipeline data
│   ├── pdfs/                  # Input: Place your PDF files here
│   ├── images/                # Generated: Page images (600 DPI)
│   ├── detections/            # Generated: YOLO detection crops
│   ├── processed/             # Generated: Extracted text/tables
│   ├── chunks/                # Generated: Semantic chunks
│   ├── embeddings/            # Generated: Embedding vectors
│   ├── indexes/               # Generated: Search indexes
│   ├── vector_store/          # Generated: Qdrant vector database
│   └── cache/                 # Generated: Embedding cache
├── models/
│   └── YOLOv11/               # YOLO model weights
├── src/                       # Source code
│   ├── prompts/               # LLM prompt templates
│   └── utils/                 # Utility modules
└── experiments/               # Jupyter notebooks
```

## Quick Start

### 1. Prepare Your Data

Create the required directories and add your PDFs:

```bash
mkdir -p data/pdfs
# Copy your PDF files to data/pdfs/
```

### 2. Run the Ingestion Pipeline

Process your documents:

```bash
python -m src.run_ingestion_pipeline --vision-provider ollama
```

Pipeline options:
- `--skip-pdf`: Skip PDF to image conversion
- `--skip-yolo`: Skip YOLO detection
- `--skip-extraction`: Skip text/table extraction
- `--skip-descriptions`: Skip image descriptions
- `--skip-chunking`: Skip semantic chunking
- `--skip-embeddings`: Skip embedding generation
- `--force-reembed`: Re-embed all content (ignore cache)
- `--vision-provider`: Choose vision provider (ollama, gemini, openai, anthropic)

### 3. Build the Indexes

After ingestion, build the search indexes:

```bash
python -m src.build_indexes
```

### 4. Launch the Chat Interface

Start the Gradio chat application:

```bash
python -m src.watchComplyChat_app
```

## Pipeline Stages

| Stage | Description | Input | Output |
|-------|-------------|-------|--------|
| PDF Conversion | Convert PDFs to images at 600 DPI | `data/pdfs/*.pdf` | `data/images/*.png` |
| YOLO Detection | Detect text, tables, pictures | `data/images/*.png` | `data/detections/` |
| Content Extraction | Extract text using DocLing | `data/detections/` | `data/processed/*.md` |
| Image Description | Describe images using vision LLM | `data/detections/` | `data/processed/*.txt` |
| Chunking | Semantic text chunking | `data/processed/` | `data/chunks/` |
| Embeddings | Generate vector embeddings | `data/chunks/` | `data/embeddings/` |
| Indexing | Build search indexes | `data/embeddings/` | `data/indexes/`, `data/vector_store/` |

## Configuration

Key settings in `src/constants.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `YOLO_CONFIDENCE_THRESHOLD` | 0.2 | YOLO detection confidence |
| `YOLO_IOU_THRESHOLD` | 0.8 | YOLO IOU threshold |
| `MAX_CHUNK_SIZE` | 512 | Maximum tokens per chunk |
| `BUFFER_SIZE` | 5 | Sentence buffer for chunking |
| `TOP_K_RETRIEVED_CHILDREN` | 10 | Number of chunks to retrieve |
| `TOP_K_RERANKED_PARENTS` | 5 | Number of results after reranking |
| `EMBED_MODEL` | qwen3-embedding:0.6b | Embedding model |

Environment variables can override defaults. Create a `.env` file:

```bash
OLLAMA_HOST=http://localhost:11434
DEFAULT_LLM_MODEL=granite3.3:8b
EMBED_MODEL=qwen3-embedding:0.6b
```

## Development

```bash
# Format code
make format

# Run linter
make lint

# Format and lint
make refactor
```

## Architecture

```
PDFs → Images (600 DPI) → YOLO Detection → Content Extraction
                                                    ↓
              ← Query Interface ← Vector Index ← Embeddings ← Semantic Chunks
```

**Retrieval Flow:**
1. User query → Query rewriting (LLM)
2. Vector search on child chunks (top-10)
3. Parent document lookup via metadata
4. LLM reranking (top-5 parents)
5. Context building + streaming response

## Dependencies

Core dependencies:
- `qdrant-client`: Vector database
- `ollama`: Local LLM inference
- `ultralytics`: YOLO object detection
- `pdf2image`: PDF conversion
- `gradio`: Web interface
- `tiktoken`: Token counting

## License

MIT
