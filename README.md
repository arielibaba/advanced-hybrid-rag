# advanced-hybrid-rag Documentation

*Generated on 2025-12-31 16:54*


| Metric | Count |
|--------|-------|
| Modules | 19 |
| Classes | 2 |
| Functions | 87 |
| Methods | 3 |
| Doc Coverage | 37.8% |


## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Domain Modules](#domain-modules)
3. [Module Reference](#module-reference)
4. [Security Considerations](#security-considerations)

---

## Architecture Overview

### Pattern: Custom Hybrid RAG Pipeline

This project implements a custom architecture tailored for hybrid RAG (Retrieval-Augmented Generation) with multi-modal document processing. The system processes PDFs through object detection, content extraction, semantic chunking, and provides a conversational interface for document querying.

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ADVANCED HYBRID RAG SYSTEM                               │
└─────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────────┐
│                              INGESTION PIPELINE                                   │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   PDF Files                                                                       │
│      │                                                                            │
│      ▼  [pdf2image @ 600 DPI]                                                    │
│   PNG Images                                                                      │
│      │                                                                            │
│      ▼  [YOLOv11 + DetectionProcessor]                                           │
│   ┌──────────────────────────────────────────────────────────────┐               │
│   │              DETECTIONS GROUPEES                              │               │
│   ├────────────────┬────────────────┬────────────────────────────┤               │
│   │ *_Text.jpg     │ *_Table_*.jpg  │ *_Picture_*.jpg            │               │
│   │      │         │       │        │        │                   │               │
│   │      ▼         │       ▼        │        ▼                   │               │
│   │ SmolDocling    │  SmolDocling   │  qwen2.5vl:7b              │               │
│   │ (MLX)          │  (MLX)         │  (Ollama Vision)           │               │
│   │      │         │       │        │        │                   │               │
│   │      ▼         │       ▼        │        ▼                   │               │
│   │  *_Text.md     │ *_Table_*.md   │ *_description.txt          │               │
│   └────────┬───────┴───────┬────────┴────────┬───────────────────┘               │
│            │               │                 │                                    │
│            └───────────────┼─────────────────┘                                    │
│                            ▼                                                      │
│   ┌──────────────────────────────────────────────────────────────┐               │
│   │                    CHUNKING LAYER                            │               │
│   ├──────────────────────────────────────────────────────────────┤               │
│   │  Text:        SemanticChunker (LangChain + Ollama)           │               │
│   │               → fallback: hard_split()                       │               │
│   │                                                              │               │
│   │  Tables:      LLM summaries + overlap chunks                 │               │
│   │               → granite3.3:8b                                │               │
│   │                                                              │               │
│   │  Descriptions: hard_split() direct                           │               │
│   │                                                              │               │
│   │  Structure:   parent_chunk ← [child_1, child_2, ...]         │               │
│   └──────────────────────────────────────────────────────────────┘               │
│                            │                                                      │
│                            ▼  [ollama.embeddings - mxbai-embed-large]            │
│   ┌──────────────────────────────────────────────────────────────┐               │
│   │  EMBEDDINGS (1,024 dimensions + metadata)                    │               │
│   │  {embedding: [...], metadata: {child, parent, source}}       │               │
│   └──────────────────────────────────────────────────────────────┘               │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE & INDEXING                                      │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   ┌─────────────────────────────────┐  ┌─────────────────────────────────┐        │
│   │    QDRANT (Vector Store)        │  │   LlamaIndex (Persistent)       │        │
│   ├─────────────────────────────────┤  ├─────────────────────────────────┤        │
│   │                                 │  │                                 │        │
│   │  child_documents                │  │  child_documents/               │        │
│   │  ├─ VectorStoreIndex            │  │  ├─ docstore.json               │        │
│   │  ├─ Distance: COSINE            │  │  ├─ vector_store.json           │        │
│   │  └─ Retrieval: top-10           │  │  └─ index_store.json            │        │
│   │                                 │  │                                 │        │
│   │  parent_documents               │  │  parent_documents/              │        │
│   │  ├─ KeywordTableIndex           │  │  ├─ docstore.json               │        │
│   │  └─ Metadata lookups            │  │  └─ index_store.json            │        │
│   │                                 │  │                                 │        │
│   └─────────────────────────────────┘  └─────────────────────────────────┘        │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE & RAG CHAT                                     │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  User Query                                                                       │
│      │                                                                            │
│      ▼  [granite3.3:8b - rewrite_query]                                          │
│  Rewritten Multi-Questions (bullet points)                                        │
│      │                                                                            │
│      ▼  [parse_rewritten_query]                                                  │
│  Individual Queries [q1, q2, q3...]                                              │
│      │                                                                            │
│      ▼  [VectorIndexRetriever - COSINE similarity]                               │
│  Retrieved Children [top-10] + similarity scores                                 │
│      │                                                                            │
│      ▼  [retrieve_from_keyword_index]                                            │
│  Parent Documents [linked via metadata]                                          │
│      │                                                                            │
│      ▼  [deduplicate by parent name]                                             │
│  Unique Parent Candidates                                                        │
│      │                                                                            │
│      ▼  [LLMRerank - granite3.3:8b]                                              │
│  Reranked Parents [top-5 by relevance]                                           │
│      │                                                                            │
│      ▼  [context_building + references extraction]                               │
│  Final Context + Source References [top-3 unique]                                │
│      │                                                                            │
│      ▼  [granite3.3:8b - streaming generation]                                   │
│  Generated Response + References                                                 │
│                                                                                   │
│   ┌─────────────────────────────────────────────────────────────────┐            │
│   │                    GRADIO UI                                    │            │
│   ├─────────────────────────────────────────────────────────────────┤            │
│   │  Chatbot (streaming, markdown)                                  │            │
│   │  User Input Textbox                                             │            │
│   │  Submit Button (queue=True)                                     │            │
│   │  Reset Button                                                   │            │
│   └─────────────────────────────────────────────────────────────────┘            │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL SERVICES (LOCAL)                               │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   OLLAMA (http://localhost:11434)                                                │
│   ├─────────────────────────────────────────────────────────────────────────┐    │
│   │                                                                         │    │
│   │  qwen2.5vl:7b          → Vision (image descriptions)                    │    │
│   │     └─ Temp: 0.2, Top_p: 0.85                                           │    │
│   │                                                                         │    │
│   │  granite3.3:8b         → Generation, Rewrite, Expand, Rerank            │    │
│   │     └─ Temp: 0.7, Top_p: 0.85, Context: 128K                            │    │
│   │                                                                         │    │
│   │  mxbai-embed-large     → Embeddings (1,024 dimensions)                  │    │
│   │                                                                         │    │
│   └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
│   SmolDocling (MLX - Apple Silicon optimized)                                    │
│   └─ ds4sd/SmolDocling-256M-preview-mlx-bf16                                     │
│                                                                                   │
│   YOLOv11x (Object Detection)                                                    │
│   └─ Conf: 0.2, IOU: 0.8                                                         │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Pipeline

| Stage | Input Directory | Output Directory | Module |
|-------|----------------|------------------|--------|
| 1. PDF Conversion | `data/pdfs/*.pdf` | `data/images/*.png` | `convert_pdf_to_image.py` |
| 2. Object Detection | `data/images/*.png` | `data/detections/*_{Text,Table,Picture}.*` | `extract_objects_from_image.py` |
| 3. Content Extraction | `data/detections/*` | `data/processed/*.md, *.txt` | `parse_image_with_*.py`, `create_image_description.py` |
| 4. Chunking | `data/processed/*` | `data/chunks/*_parent_*_child_*.txt` | `chunk_text_data.py`, `chunk_table_data.py` |
| 5. Embeddings | `data/chunks/*` | `data/embeddings/*_embedding.json` | `create_embeddings.py` |
| 6. Indexing | `data/embeddings/*` | `data/vector_store/`, `data/indexes/` | `build_indexes.py` |

### Key Configuration (constants.py)

| Category | Parameter | Value |
|----------|-----------|-------|
| **YOLO** | Confidence | 0.2 |
| **YOLO** | IOU | 0.8 |
| **Chunking** | Max tokens | 512 |
| **Chunking** | Buffer size | 5 |
| **Retrieval** | Children retrieved | top-10 |
| **Retrieval** | Parents after rerank | top-5 |
| **LLM** | Context window | 128K |
| **LLM** | Memory window | 64K |
| **Ollama** | Timeout | 180s |

### Layers

- **Data Ingestion**: PDF conversion to high-resolution images (600 DPI) and YOLO-based object detection (tables, text, pictures).

- **Content Extraction**: SmolDocling MLX for text/table OCR, Vision LLM (qwen2.5vl) for image descriptions.

- **Data Processing & Chunking**: Semantic chunking with LangChain + Ollama embeddings, hierarchical parent-child structure.

- **Embedding & Indexing**: Vector embeddings (mxbai-embed-large), hybrid storage with Qdrant (vectors) and LlamaIndex (keywords).

- **RAG Inference**: Multi-step retrieval with query rewriting, vector search, parent lookup, LLM reranking, and streaming generation.

- **Utilities**: Logging, text manipulation, file handling, and API interactions.

### Strengths

- **Multi-modal Processing**: Handles text, tables, and images from PDFs with specialized extraction pipelines.

- **Hierarchical Chunking**: Parent-child relationship enables context-aware retrieval.

- **Hybrid Retrieval**: Combines vector similarity (children) with keyword lookup (parents) for better precision.

- **LLM Reranking**: Improves retrieval quality by reranking with the generation model.

- **Local Inference**: All models run locally via Ollama for privacy and control.

- **Streaming UI**: Real-time response generation with Gradio interface.

### Architectural Concerns

- **[MEDIUM]** Tight coupling between modules if not designed carefully: Ensure loose coupling between modules by using interfaces or abstract classes where appropriate.

- **[LOW]** Limited information on error handling and monitoring: Implement robust error handling and monitoring mechanisms to ensure the pipeline's reliability.


---

## Domain Modules


*10 functional domains identified:*


### Text Chunking and Token Management

This module focuses on splitting text into manageable chunks, primarily for use in language models or systems with token limits. It provides functionalities for semantic chunking, hard splitting based on token count, and managing chat history within token constraints.


**Key Components:**

- `chunk_text_data`: Module

- `hard_split`: Function: Simple text splitter

- `process_semantic_split`: Function: Semantic text splitter

- `semantic_chunk_text_file`: Function: Processes text files into semantic chunks

- `get_token_count`: Function: Counts tokens in text

- `limit_chat_history`: Function: Truncates chat history based on token limit


**Public API:**

- `hard_split`

- `process_semantic_split`

- `semantic_chunk_text_file`

- `chunk_text_data`

- `get_token_count`

- `limit_chat_history`


*Cohesion: high*


### Embedding Generation

This component focuses on generating and managing text embeddings. It reads text chunks, creates embeddings using a specified model, and stores the embeddings along with metadata for later retrieval and use in semantic search or other NLP tasks.


**Key Components:**

- `create_embeddings`: Module responsible for generating embeddings for text chunks.

- `get_embeddings`: Function to generate an embedding for a given text.

- `make_metadata`: Function to build metadata dictionary based on chunk filename.


**Public API:**

- `create_embeddings.get_embeddings`

- `create_embeddings.make_metadata`

- `create_embeddings.create_embeddings`


*Cohesion: high*


### Markdown Table Chunking and LLM Interaction

This cluster of code entities focuses on processing markdown tables, splitting them into manageable chunks, and interacting with a Large Language Model (LLM) to generate summaries or responses based on the table content. It handles potential malformed JSON responses from the LLM and extracts valid JSON.


**Key Components:**

- `chunk_table_data`: Module containing table chunking functions.

- `chunk_markdown_table_with_overlap`: Function to split markdown tables into overlapping chunks.

- `chunk_markdown_table`: Function to split markdown tables into chunks and generate summaries using an LLM.

- `chunk_table_data`: Function to open markdown tables and split them into chunks.

- `ask_LLM_with_JSON`: Function to query an LLM and parse the JSON response.

- `recover_json`: Function to recover valid JSON from potentially malformed strings.

- `extract_json`: Function to extract JSON from a larger string.


**Public API:**

- `chunk_markdown_table_with_overlap`

- `chunk_markdown_table`

- `chunk_table_data`

- `ask_LLM_with_JSON`

- `recover_json`

- `extract_json`


*Cohesion: high*


### Document Object Detection and Processing

This code cluster focuses on detecting and grouping objects (tables, pictures, and other detections) within an image. It involves running inference, grouping detections based on spatial relationships (specifically vertical overlap), and saving the grouped and remaining detections.


**Key Components:**

- `extract_objects_from_image`: Module containing functions for object extraction.

- `vertical_overlap`: Function to determine vertical overlap between objects.

- `DetectionProcessor`: Class responsible for processing and grouping detections.

- `process_image`: Method to run inference and group detections.

- `save_detections`: Method to save grouped and remaining detections.


**Public API:**

- `DetectionProcessor`

- `process_image`

- `save_detections`

- `extract_objects_from_image`

- `vertical_overlap`


*Cohesion: high*


### Image Description Generation

Asynchronously generates descriptions for relevant images within a directory. It filters images based on relevance criteria (non-white pixel count and color variance), creates descriptions using an external language model, and extracts Markdown tables from the generated descriptions.


**Key Components:**

- `create_image_descriptions_async`: Orchestrates the asynchronous image description generation process.

- `_describe_single_image`: Generates a description for a single image using a language model.

- `is_relevant_image`: Determines if an image is relevant based on pixel analysis.

- `extract_markdown_tables`: Extracts Markdown tables from text.


**Public API:**

- `create_image_descriptions_async`


*Cohesion: high*


### Question Answering and Document Processing

This cluster of code entities focuses on processing and querying documents. It includes functionalities for cleaning text, creating and managing indexes, rewriting and expanding queries, and interacting with Language Model Models (LLMs) for generating responses. The code supports both streaming and non-streaming modes of LLM interaction.


**Key Components:**

- `helpers`: Module containing utility functions for text manipulation.

- `markdown_to_plain_text`: Function to convert Markdown to plain text.

- `remove_markdown_tables`: Function to remove Markdown tables from text.

- `remove_code_blocks`: Function to remove code blocks from text.

- `remove_mermaid_blocks`: Function to remove Mermaid diagrams from text.

- `remove_extracted_text_blocks`: Function to remove extracted text blocks from text.

- `clean_up_text`: Function to clean up text by removing code blocks, Mermaid diagrams, and Markdown tables.

- `load_embeddings_with_associated_documents`: Function to load embeddings and associated documents.


**Public API:**

- `markdown_to_plain_text`

- `clean_up_text`

- `load_embeddings_with_associated_documents`

- `create_documents`

- `create_index`

- `save_index`

- `query_index`

- `retrieve_from_keyword_index`


*Cohesion: high*


### Text Processing and Logging Utilities

This cluster provides a collection of utility functions for text manipulation, extraction, and logging. It includes functionalities for extracting specific content types (JSON, SQL, code, markdown, tables, etc.) from text, managing token counts, cleaning text, and logging events.


**Key Components:**

- `logc`: Module for logging functionalities

- `text_utils`: Module for text manipulation and extraction

- `get_token_count`: Function to calculate token count in a text

- `limit_token_count`: Function to limit text based on token count

- `extract_json`: Function to extract JSON from text

- `extract_sql`: Function to extract SQL from text

- `extract_markdown_table_as_df`: Function to extract markdown table as a Pandas DataFrame


**Public API:**

- `get_current_time`

- `logc`

- `show_json`

- `get_encoder`

- `get_token_count`

- `limit_token_count`

- `extract_json`

- `extract_sql`


*Cohesion: high*


### OpenAI API Interaction

This module provides a set of utility functions for interacting with the OpenAI API. It encapsulates functionalities such as generating chat completions, creating embeddings, and handling image-based interactions with GPT-4 Vision, including format conversions and base64 encoding.


**Key Components:**

- `get_chat_completion`: Function for generating chat completions from OpenAI's models.

- `get_chat_completion_with_json`: Function for generating chat completions with JSON output from OpenAI's models.

- `get_embeddings`: Function for creating embeddings using OpenAI's models.

- `ask_LLM`: Function to ask a Large Language Model (LLM) a question and get a response.

- `ask_LLM_with_JSON`: Function to ask an LLM a question and get a JSON response.

- `get_image_base64`: Function to encode an image to base64 format.

- `convert_png_to_jpg`: Function to convert a PNG image to JPG format.

- `call_gpt4v`: Function to interact with GPT-4 Vision, handling image processing and JSON recovery.


**Public API:**

- `get_chat_completion`

- `get_chat_completion_with_json`

- `get_embeddings`

- `ask_LLM`

- `ask_LLM_with_JSON`

- `get_image_base64`

- `convert_png_to_jpg`

- `call_gpt4v`


*Cohesion: high*


### File System Utilities

Provides a collection of utility functions for interacting with the file system. This includes functionalities for checking file existence, saving and loading data using pickle, manipulating file extensions, reading and writing file content, and finding files based on specific criteria.


**Key Components:**

- `file_utils`: Module containing file system utilities

- `is_file_or_url`: Function to check if a path is a file or URL

- `save_to_pickle`: Function to save data to a pickle file

- `load_from_pickle`: Function to load data from a pickle file

- `check_replace_extension`: Function to check and replace file extension

- `replace_extension`: Function to replace file extension

- `write_to_file`: Function to write content to a file

- `read_asset_file`: Function to read content from an asset file


**Public API:**

- `is_file_or_url`

- `save_to_pickle`

- `load_from_pickle`

- `check_replace_extension`

- `replace_extension`

- `write_to_file`

- `read_asset_file`

- `find_certain_files`


*Cohesion: high*


### Terminal Output Formatting

Provides a set of color codes to format terminal output. This allows developers to add visual cues and improve the readability of console messages by using different colors and styles.


**Key Components:**

- `bcolors`: Container for color codes


**Public API:**

- `bcolors class (containing color code attributes)`


*Cohesion: high*


---

## Module Reference


### Package: `Root`


#### `__init__`

*File: __init__.py*


#### `build_indexes`

*File: build_indexes.py*


#### `chunk_table_data`

*File: chunk_table_data.py*


**Functions:**

- `chunk_markdown_table_with_overlap`: Splits a markdown table into chunks with overlapping tokens.

- `chunk_markdown_table`: Splits a markdown table into chunks with overlapping tokens and generates a summary of the table.

- `chunk_table_data`: Opens markdown tables and split them into chunnks, then, stored the resulting tables chunks as well 


#### `chunk_text_data`

*File: chunk_text_data.py*


**Functions:**

- `hard_split`: Fallback simple splitter that breaks a text into chunks

- `process_semantic_split`: Splits a list of texts semantically into chunks within the token limit.

- `semantic_chunk_text_file`: Processes a single text file and returns semantic chunks.

- `chunk_text_data`: Performs txt documents chunking using semantic chunking. Then, stores the resulting chunks into a de


#### `constants`

*File: constants.py*


#### `convert_pdf_to_image`

*File: convert_pdf_to_image.py*


**Functions:**

- `convert_pdf_to_image`: Converts each page of every PDF file in the specified directory into separate image files.


#### `create_embeddings`

*File: create_embeddings.py*


**Functions:**

- `get_embeddings`: Genetrates an embedding for a given text.

- `make_metadata`: Build the metadata dict based on the chunk filename.

- `create_embeddings`: Generates embeddings for all chunk files in `chunks_dir`


#### `create_image_description`

*File: create_image_description.py*


**Functions:**

- `create_image_descriptions_async`: Describe all *_Picture_*.jpg in `image_dir` with a bounded semaphore.


#### `extract_objects_from_image`

*File: extract_objects_from_image.py*


**Classes:**


##### `DetectionProcessor`

The `DetectionProcessor` class is responsible for detecting objects (specifically tables and pictures) within images using a YOLO model. It groups rel...


| Method | Description |
|--------|-------------|

| `process_image` | Run inference on the given image and perform detection grouping. |

| `save_detections` | Save each Table/Picture group separately and composite all remaining detections  |


**Functions:**

- `vertical_overlap`: 

- `extract_objects_from_image`: 


#### `helpers`

*File: helpers.py*


**Functions:**

- `clear_pytorch_cache`: Clears MPS cache in PyTorch if available.

- `load_prompt`: 

- `is_relevant_image`: Determines if an image is relevant based on non-white pixel count and color variance.

- `markdown_to_plain_text`: Converts Markdown text to plain text by first converting it to HTML

- `ask_LLM_with_JSON`: Asks the LLM to generate a response based on the provided prompt.

- `recover_json`: "

- `extract_json`: Extracts a JSON string from a larger string that may contain Markdown formatting.

- `extract_markdown_tables`: Finds all Markdown tables in the text.

- `remove_markdown_tables`: Removes all Markdown tables (as defined above) from the text.

- `remove_code_blocks`: Strips out any fenced code block (```...```), regardless of language.

- `remove_mermaid_blocks`: Specifically strips out any ```mermaid ... ``` block.

- `remove_extracted_text_blocks`: Removes any ```EXTRACTED TEXT ... ``` sections.

- `clean_up_text`: Cleans up the input text by:

- `get_token_count`: Returns the number of tokens for the input text.

- `load_embeddings_with_associated_documents`: Loads the embeddings and the associated source documents (children and parents) from corresponding d

- `create_documents`: 

- `create_index`: Create an index from a given set of documents and a storage context.

- `save_index`: Saves the given index to the specified directory.

- `query_index`: Queries the given index with the provided query string.

- `retrieve_from_keyword_index`: Return a list of Documents whose metadata[key] == value.

- `limit_chat_history`: Truncates the chat history to fit within a maximum token limit.

- `stream_llm_output`: Streams the output of an LLM chat response.

- `run_streaming`: Runs the LLM in streaming mode and yields the response as it is generated.

- `rewrite_query`: Rewrites the user query based on the conversation history and the new question.

- `parse_rewritten_query`: Extracts bullet-pointed questions from the rewritten query.

- `expand_query`: Expands the user query by identifying synonyms or related terms using the LLM.

- `parse_expanded_queries`: Parses the expanded queries from the LLM response text.

- `convert_history_to_tuples`: Converts a chat history list of dictionaries into a list of tuples for Gradio.

- `reset_conversation`: Resets the conversation history and returns an empty list and an empty string.


#### `parse_image_with_table`

*File: parse_image_with_table.py*


**Functions:**

- `parse_image_with_table_func`: 

- `parse_image_with_table`: 


#### `parse_image_with_text`

*File: parse_image_with_text.py*


**Functions:**

- `parse_image_with_text_func`: 

- `parse_image_with_text`: 


#### `run_ingestion_pipeline`

*File: run_ingestion_pipeline.py*


**Functions:**

- `run_ingestion_pipeline_async`: 


### Package: `utils`


#### `__init__`

*File: utils/__init__.py*


#### `bcolors`

*File: utils/bcolors.py*


**Classes:**


##### `bcolors`

Provides a set of ANSI escape codes for formatting terminal output with colors and styles, enhancing readability and user experience in command-line a...


#### `file_utils`

*File: utils/file_utils.py*


**Functions:**

- `is_file_or_url`: 

- `save_to_pickle`: 

- `load_from_pickle`: 

- `check_replace_extension`: 

- `replace_extension`: 

- `write_to_file`: 

- `read_asset_file`: 

- `find_certain_files`: 


#### `logc`

*File: utils/logc.py*


**Functions:**

- `get_current_time`: 

- `logc`: 


#### `openai_utils`

*File: utils/openai_utils.py*


**Functions:**

- `get_chat_completion`: 

- `get_chat_completion_with_json`: 

- `get_embeddings`: 

- `ask_LLM`: 

- `ask_LLM_with_JSON`: 

- `get_image_base64`: 

- `convert_png_to_jpg`: 

- `call_gpt4v`: 


#### `text_utils`

*File: utils/text_utils.py*


**Functions:**

- `show_json`: 

- `get_encoder`: 

- `get_token_count`: 

- `limit_token_count`: 

- `extract_json`: 

- `extract_sql`: 

- `extract_code`: 

- `extract_extracted_text`: 

- `extract_markdown`: 

- `extract_mermaid`: 

- `extract_markdown_table`: 

- `extract_table_rows`: 

- `extract_markdown_table_as_df`: 

- `remove_code`: 

- `remove_markdown`: 

- `remove_mermaid`: 

- `remove_extracted_text`: 

- `clean_up_text`: 

- `recover_json`: 

- `extract_chunk_number`: 


---

## API Reference

No API endpoints detected.


---

## Security Considerations


*3 potential security concerns identified (Critical/High only):*


### [HIGH] Arbitrary Code Execution

**Location:** `semantic_chunk_text_file`


The `embed_model_name` parameter is directly passed to `OllamaEmbeddings`. If an attacker can control this parameter, they could potentially execute arbitrary code by specifying a malicious model name


**Recommendation:** Sanitize and validate the `embed_model_name` and `sentence_split_regex` parameters. Implement a whitelist of allowed model names for `embed_model_name`.  Implement input validation and complexity limi


### [HIGH] Credential Exposure

**Location:** `ask_LLM`


The code directly uses API keys (`model_info['AZURE_OPENAI_KEY']`) from `model_info` which could be sourced from environment variables or configuration files. If `model_info` is not properly secured o


**Recommendation:** Store and retrieve API keys securely using a secrets management system (e.g., Azure Key Vault) and ensure proper access controls are in place. Avoid hardcoding API keys directly in the code.


### [HIGH] API Key Exposure

**Location:** `ask_LLM_with_JSON`


The code directly uses the `AZURE_OPENAI_KEY` from `model_info`. If `model_info` comes from an untrusted source (e.g., user input, configuration file without proper access control), it could lead to e


**Recommendation:** Store the API key securely using environment variables or a secrets management system. Ensure that access to the secrets is properly controlled and restricted to authorized personnel or services. Avoi
