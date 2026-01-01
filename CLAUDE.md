# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Advanced Hybrid RAG is a document processing and retrieval pipeline that:
- Converts PDFs to images and detects objects (tables, text, pictures) using YOLO
- Parses content using SmolDocling model
- Generates semantic chunks and embeddings
- Builds vector indexes for similarity search
- Provides a Gradio chat interface for document querying

## Build and Development Commands

```bash
# Package management (uses uv)
make install          # Install dependencies
make sync             # Sync environment with lock file
make lock             # Lock dependencies

# Code quality
make format           # Format code with black
make lint             # Run pylint on source
make refactor         # Format and lint

# Run the pipeline
python src/run_ingestion_pipeline.py

# Launch chat interface
python src/watchComplyChat_app.py
```

## Architecture

### Pipeline Stages

```
PDFs → Images (600 DPI) → YOLO Detection → Text/Table/Image Extraction
→ Semantic Chunking → Embeddings → Vector Index → Query Interface
```

### Key Modules

- **`src/run_ingestion_pipeline.py`**: Main pipeline orchestrator (async)
- **`src/watchComplyChat_app.py`**: Gradio chat application with query rewriting, expansion, and LLM reranking
- **`src/constants.py`**: All configuration values and path definitions
- **`src/extract_objects_from_image.py`**: `DetectionProcessor` class for YOLO inference and grouping
- **`src/create_image_description.py`**: Async image processing with semaphore-limited concurrency
- **`src/chunk_text_data.py`**: Semantic chunking using LangChain with OllamaEmbeddings
- **`src/build_indexes.py`**: LlamaIndex + Qdrant vector store with hierarchical retrieval

### Data Flow

| Stage | Input Directory | Output Directory |
|-------|----------------|------------------|
| PDF Conversion | `data/pdfs/` | `data/images/` |
| YOLO Detection | `data/images/` | `data/detections/` |
| Content Extraction | `data/detections/` | `data/processed/` |
| Chunking | `data/processed/` | `data/chunks/` |
| Embeddings | `data/chunks/` | `data/embeddings/` |
| Indexing | `data/embeddings/` | `data/indexes/`, `data/vector_store/` |

### External Services

All inference runs locally via Ollama at `http://localhost:11434`:
- **LLM**: granite3.3:8b (128K context)
- **Vision**: qwen2.5vl:7b
- **Embeddings**: mxbai-embed-large

## Configuration

Key settings in `src/constants.py`:
- YOLO thresholds: confidence=0.2, IOU=0.8
- Chunking: max 512 tokens, buffer 5 tokens
- Retrieval: top-10 children → LLM rerank → top-5 parents
- Ollama timeout: 180 seconds

## Prompt Templates

All prompts are in `src/prompts/` as markdown files, covering:
- Image/text/table extraction
- Query rewriting and expansion
- Final answer generation

## Notebooks

Jupyter notebooks in `experiments/`:
- `run_pipeline.ipynb`: Step-by-step pipeline execution
- `run_functions.ipynb`: Individual function testing
- `launch_webapp.ipynb`: Web app launcher
