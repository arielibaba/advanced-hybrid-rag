import os
import asyncio
import logging

import ollama

from .constants import (
    PDF_DIR,
    IMAGE_DIR,
    DETECTION_DIR,
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    SMOLDOCLING_MODEL_DIR,
    PROCESSED_DIR,
    CHUNKS_DIR,
    EMBED_MODEL,
    MAX_CHUNK_SIZE,
    BUFFER_SIZE,
    BREAKPOINT_THRESHOLD_TYPE,
    BREAKPOINT_THRESHOLD_AMOUNT,
    SENTENCE_SPLIT_REGEX,
    SUMMARIZE_TABLE_PROMPT,
    EMBEDDINGS_DIR,
    VLLM,
    LLM,
    TEMPERATURE_IMAGE_DESC,
    TOP_P_IMAGE_DESC,
    TEMPERATURE_GENERATION,
    TOP_P_GENERATION,
    SYSTEM_PROMPT_IMAGE_DESC,
    USER_PROMPT_IMAGE_DESC
)

from .helpers import clear_pytorch_cache, load_prompt
from .convert_pdf_to_image import convert_pdf_to_image
from .extract_objects_from_image import extract_objects_from_image
from .parse_image_with_text import parse_image_with_text
from .parse_image_with_table import parse_image_with_table
from .create_image_description import create_image_descriptions_async
from .chunk_text_data import chunk_text_data
from .chunk_table_data import chunk_table_data
from .create_embeddings import create_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


async def run_ingestion_pipeline_async():
    # 1. Clear GPU memory if any
    clear_pytorch_cache()

    # 2. Convert PDF pages → images
    logging.info("Converting PDFs to images…")
    convert_pdf_to_image(PDF_DIR, IMAGE_DIR)

    # 3. Detect objects (text, tables, pics…) in images
    logging.info("Running YOLO on images…")
    extract_objects_from_image(
        IMAGE_DIR,
        DETECTION_DIR,
        YOLO_MODEL_PATH,
        YOLO_CONFIDENCE_THRESHOLD,
        YOLO_IOU_THRESHOLD,
        high_quality=True
    )

    # 4. Extract text and tables from those crops
    logging.info("Parsing text from images…")
    parse_image_with_text(
        DETECTION_DIR,
        SMOLDOCLING_MODEL_DIR,
        "Find all 'text' elements on the page, retrieve all section headers.",
        PROCESSED_DIR
    )
    logging.info("Parsing tables from images…")
    parse_image_with_table(
        DETECTION_DIR,
        SMOLDOCLING_MODEL_DIR,
        "Detect the table elements on the page, use OCR when needed.",
        PROCESSED_DIR
    )

    # 5. Describe pictures via vision-LLM in parallel
    print()
    logging.info("Generating image descriptions…")
    print()
    client = ollama.Client()
    system_p = load_prompt(SYSTEM_PROMPT_IMAGE_DESC)
    user_p   = load_prompt(USER_PROMPT_IMAGE_DESC)
    model_opts = {
        "temperature": TEMPERATURE_IMAGE_DESC,
        "top_p": TOP_P_IMAGE_DESC
    }
    await create_image_descriptions_async(
        image_dir=DETECTION_DIR,
        ollama_client=client,
        model=VLLM,
        system_prompt=system_p,
        description_prompt=user_p,
        model_options=model_opts,
        output_dir=PROCESSED_DIR,
        max_concurrency=5
    )

    # 6. Chunk the resulting text
    logging.info("Chunking text data…")
    chunk_text_data(
        PROCESSED_DIR,
        EMBED_MODEL,
        MAX_CHUNK_SIZE,
        None,
        CHUNKS_DIR,
        BUFFER_SIZE,
        BREAKPOINT_THRESHOLD_TYPE,
        BREAKPOINT_THRESHOLD_AMOUNT,
        SENTENCE_SPLIT_REGEX,
        verbose=True
    )

    # 7. Chunk any extracted tables
    logging.info("Chunking table data…")
    with open(SUMMARIZE_TABLE_PROMPT, encoding="utf-8") as f:
        table_prompt = f.read()
    chunk_table_data(
        table_prompt,
        PROCESSED_DIR,
        None,
        MAX_CHUNK_SIZE,
        int(0.25 * MAX_CHUNK_SIZE),
        client,
        LLM,
        {"temperature": TEMPERATURE_GENERATION, "top_p": TOP_P_GENERATION},
        CHUNKS_DIR
    )

    # 8. Build embeddings for all chunks
    logging.info("Creating embeddings…")
    create_embeddings(CHUNKS_DIR, EMBEDDINGS_DIR, EMBED_MODEL)

    logging.info("Ingestion pipeline complete.")


if __name__ == "__main__":
    asyncio.run(run_ingestion_pipeline_async())