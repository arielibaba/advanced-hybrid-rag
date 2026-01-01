from pathlib import Path

# Get the directory of the current file (constants.py)
BASE_DIR = Path(__file__).resolve().parent

# Define paths relative to the location of this file
PDF_DIR = BASE_DIR / "../data/pdfs"
PDF_CONVERTED_DIR = BASE_DIR / "../data/pdfs_converted"
IMAGE_DIR = BASE_DIR / "../data/images"
DETECTION_DIR = BASE_DIR / "../data/detections"
PROCESSED_DIR = BASE_DIR / "../data/processed"

# Handled file extensions
FILE_EXTENSIONS = ['.doc', '.docx', '.html', '.htm', '.ppt', '.pptx']
FILE_EXTENSIONS.extend([ext.upper() for ext in FILE_EXTENSIONS])

# Yolo Model Configuration
YOLO_MODEL_PATH = BASE_DIR / "../models/YOLOv11/yolov11x_best.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.2
YOLO_IOU_THRESHOLD = 0.8

# Parameters to Use for Filtering the Extracted Images
IMAGE_PIXEL_THRESHOLD = 10000
IMAGE_PIXEL_VARIANCE_THRESHOLD = 500

# SmolDocling Model Configuration
SMOLDOCLING_MODEL_DIR = BASE_DIR / "../models/ds4sd_SmolDocling-256M-preview-mlx-bf16"

OLLAMA_URL = "http://localhost:11434"  # Ollama server URL, adjust if needed
OLLAMA_REQUEST_TIMEOUT = 180.0  # Timeout for Ollama requests in seconds

# Ollama Vision Model Configuration
VLLM = "qwen2.5vl:7b" #"llama3.2-vision:11b", "gemma3:12b"
TEMPERATURE_IMAGE_DESC = 0.2
TEMPERATURE_TEXT_EXTRACT = 0.1
TOP_P_IMAGE_DESC = 0.85
TOP_P_IMAGE_TEXT_EXTRACT = 0.1

# Ollama Thinking Model Configuration
LLM = "granite3.3:8b"        # qwen3 (40K): "qwen3:8b-q8_0", "qwen3:14b"
                             # phi4: "phi4-mini:3.8b" (128K), "phi4-mini:3.8b-fp16 " (4K), "phi4-mini-reasoning" (4K), "phi4-mini-reasoning:3.8b-fp16" (4K), "phi4:14b" (16K)
                             # "granite3.3:8b" (128K),
                             # "deepseek-r1:14b" (128K)
TEMPERATURE_GENERATION = 0.7
TOP_P_GENERATION = 0.85
CONTEXT_WINDOW = 128e3                   # LLM contextlength window, adjust based on the model used, e.g. 128k for Qwen2.5VL, 32k for Llama3.2, etc.
MEMORY_WINDOW = CONTEXT_WINDOW * 0.5    # Maximum lenght of the memory to keep track over the conversation - can start with 50% of the context window of the LLM for example

# Query Expansion
TEMPERATURE_QUERY_EXPANSION = 0.3       # query expansion task by LLM (e.g. HyDE, etc)

# Prompts Files
SYSTEM_PROMPT_IMAGE_DESC = BASE_DIR / "prompts/system_prompt_for_image_description.md"
USER_PROMPT_IMAGE_DESC = BASE_DIR / "prompts/user_prompt_for_image_description.md"

SYSTEM_PROMPT_TEXT_EXTRACT = BASE_DIR / "prompts/system_prompt_for_text_extract.md"
USER_PROMPT_TEXT_EXTRACT = BASE_DIR / "prompts/user_prompt_for_text_extract.md"

SUMMARIZE_TABLE_PROMPT = BASE_DIR / "prompts/markdown_extract_header_and_summarize_prompt.md"

SYSTEM_PROMPT_REWRITE_QUERY = BASE_DIR / "prompts/system_prompt_rewrite_query.md"
USER_PROMPT_REWRITE_QUERY = BASE_DIR / "prompts/user_prompt_rewrite_query.md"

SYSTEM_PROMPT_EXPAND_QUERY = BASE_DIR / "prompts/system_prompt_expand_query.md"
USER_PROMPT_EXPAND_QUERY = BASE_DIR / "prompts/user_prompt_expand_query.md"

SYSTEM_PROMPT_GENERATE_FINAL_ANSWER = BASE_DIR / "prompts/system_prompt_generate_final_answer.md"
USER_PROMPT_GENERATE_FINAL_ANSWER = BASE_DIR / "prompts/user_prompt_generate_final_answer.md"

# Chunking
CHUNKS_DIR = BASE_DIR / "../data/chunks"
EMBED_MODEL = "mxbai-embed-large"
MAX_CHUNK_SIZE = 512  # Maximum size of each chunk in tokens, let's say 512 tokens for parent chunks and 64 for child chunks
BUFFER_SIZE = 5  # Buffer size for the semantic chunker
BREAKPOINT_THRESHOLD_TYPE = "percentile"  # Type of breakpoint threshold
BREAKPOINT_THRESHOLD_AMOUNT = 0.95  # Amount for the breakpoint threshold
SENTENCE_SPLIT_REGEX = r"\n\n\n"  # Regex to split sentences, using triple newlines as a delimiter. This can be adjusted based on the text structure.

# Embedding
EMBEDDINGS_DIR = BASE_DIR / "../data/embeddings"

# Qdrant Vector Store
VECTOR_STORE_DIR = BASE_DIR / "../data/vector_store"


# Indexing
INDEX_DIR = BASE_DIR / "../data/indexes"
CHILD_DOCUMENTS_INDEX =  "child_documents"
PARENT_DOCUMENTS_INDEX = "parent_documents"

# Retrieval Parameters for the Search Engine
TOP_K_RETRIEVED_CHILDREN = 10           # Number of nodes retrieved
TOP_K_RERANKED_PARENTS = 5              # Number of nodes retained after reranking - can start with 5 for example
TOP_K_REFS = 3                          # Number of references to display - must be at most the number of retrieved parents

# Optional: convert all Path objects to strings if needed elsewhere
PDF_DIR = str(PDF_DIR.resolve())
PDF_CONVERTED_DIR = str(PDF_CONVERTED_DIR.resolve())
IMAGE_DIR = str(IMAGE_DIR.resolve())
DETECTION_DIR = str(DETECTION_DIR.resolve())
PROCESSED_DIR= str(PROCESSED_DIR.resolve())
YOLO_MODEL_PATH = str(YOLO_MODEL_PATH.resolve())
SYSTEM_PROMPT_IMAGE_DESC = str(SYSTEM_PROMPT_IMAGE_DESC.resolve())
SYSTEM_PROMPT_TEXT_EXTRACT = str(SYSTEM_PROMPT_TEXT_EXTRACT.resolve())
USER_PROMPT_IMAGE_DESC = str(USER_PROMPT_IMAGE_DESC.resolve())
USER_PROMPT_TEXT_EXTRACT = str(USER_PROMPT_TEXT_EXTRACT.resolve())
SMOLDOCLING_MODEL_DIR = str(SMOLDOCLING_MODEL_DIR.resolve())
CHUNKS_DIR = str((BASE_DIR / "../data/chunks").resolve())
EMBEDDINGS_DIR = str(EMBEDDINGS_DIR.resolve())
VECTOR_STORE_DIR = str(VECTOR_STORE_DIR.resolve())
INDEX_DIR = str(INDEX_DIR.resolve())
