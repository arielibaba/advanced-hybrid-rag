from pathlib import Path
from collections import deque
from typing import Dict, List
import json

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings

import ollama

from .helpers import get_token_count




def get_embeddings(text: str, embed_model: str) -> List[float]:
    """
    Genetrates an embedding for a given text.

    Args:
        - text (str): The text to embed.
        - embed_model (str): The name of the embedding model.

    Returns:
        The embedding vector as a list of floats.
    """
    response = ollama.embeddings(
        model=embed_model,
        prompt=text
    )

    embed_vector = response['embedding']  # list of ~1,024 floats
    return embed_vector

def make_metadata(chunk_filename: str) -> dict:
    """
    Build the metadata dict based on the chunk filename.
    """
    # Determine parent filename
    if "_child_chunk_" in chunk_filename:
        parent = chunk_filename.split("_child_chunk_")[0] + ".txt"
    elif "_description_" in chunk_filename:
        parent = chunk_filename.split("_chunk_")[0] + ".txt"
    elif "_Table_" in chunk_filename:
        parent = chunk_filename.split("_chunk_")[0] + ".md"
    else:
        parent = None

    # Extract document and page
    document = None
    page = None
    if "_page_" in chunk_filename:
        try:
            before, after = chunk_filename.split("_page_", 1)
            document = before
            page = after.split("_")[0]
        except ValueError:
            # filename didn’t match expected pattern
            pass

    return {
        "child": chunk_filename,
        "parent": parent,
        "source": {
            "document": document,
            "page": page
        }
    }

def create_embeddings(
    chunks_dir: str,
    embeddings_dir: str,
    embed_model: str
) -> None:
    """
    Generates embeddings for all chunk files in `chunks_dir`
    and writes them as JSON into `embeddings_dir`.
    """
    chunks_path = Path(chunks_dir)
    embeddings_path = Path(embeddings_dir)

    chunks_path.mkdir(parents=True, exist_ok=True)
    embeddings_path.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing the files in {chunks_path}...")

    for file_path in chunks_path.rglob("*"):
        # Skip directories, parent chunks, and trivially short files
        if (
            not file_path.is_file() or
            "_parent_chunk_" in file_path.name
        ):
            continue

        text = file_path.read_text(encoding="utf-8")
        if not text.strip() or len(text.split()) < 3:
            continue

        print(f"\nCalculating embedding for: {file_path.name}...")
        try:
            embed_vector = get_embeddings(text, embed_model)
            embedding_file = embeddings_path / f"{file_path.stem}_embedding.json"
        except Exception as e:
            print(f"Error generating embedding for {file_path.name}: {e}")
            continue

        # Build metadata and write out JSON
        meta = make_metadata(file_path.name)
        data = {
            "embedding": embed_vector,
            "metadata": meta
        }

        try:
            with open(embedding_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print(f"Embedding has been saved in file: {embedding_file}.")
        except Exception as e:
            print(f"Error writing embedding file {embedding_file.name}: {e}")

    print(
        f"\nAll files have been processed "
        f"and embeddings stored to: {embeddings_dir}.\n"
    )


