from pathlib import Path
from collections import deque
from typing import Dict

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings

import ollama

from .helpers import get_token_count, clean_up_text

from .constants import (
    PROCESSED_DIR,
    CHUNKS_DIR,
    EMBED_MODEL,
    MAX_CHUNK_SIZE,
    BUFFER_SIZE,
    BREAKPOINT_THRESHOLD_TYPE,
    BREAKPOINT_THRESHOLD_AMOUNT,
    SENTENCE_SPLIT_REGEX,
    OLLAMA_URL
)


from typing import List

def hard_split(
    text: str,
    max_chunk_size: int,
    overlap: int = 0
) -> List[str]:
    """
    Fallback simple splitter that breaks a text into chunks
    of at most max_chunk_size tokens, using whitespace as delimiter,
    and carries `overlap` tokens from the end of each chunk
    into the start of the next chunk.

    Args:
        text (str): The text to split.
        max_chunk_size (int): Maximum allowed size for each chunk in tokens.
        overlap (int): Number of tokens to overlap between consecutive chunks.

    Returns:
        List[str]: List of text segments that are within the token limit,
                   each (after the first) starting with the last `overlap`
                   tokens of the previous chunk.
    """
    words = text.split()
    chunks: List[str] = []
    current_words: List[str] = []

    for word in words:
        # try adding this word to the current chunk
        candidate = current_words + [word]
        if get_token_count(" ".join(candidate)) > max_chunk_size:
            # finalize the current chunk
            chunks.append(" ".join(current_words))

            # prepare the next chunk start with the last `overlap` tokens
            if overlap > 0:
                # take at most `overlap` tokens from the end
                carry = current_words[-overlap:]
            else:
                carry = []
            current_words = carry + [word]
        else:
            # safe to add
            current_words = candidate

    # add any remaining words as the last chunk
    if current_words:
        chunks.append(" ".join(current_words))

    return chunks

def process_semantic_split(texts, semantic_splitter, max_chunk_size):
    """Splits a list of texts semantically into chunks within the token limit.

    Args:
        texts (list): List of text segments to process.
        semantic_splitter (SemanticChunker): The semantic splitter instance.
        max_chunk_size (int): Maximum allowed size for each chunk in tokens.

    Returns:
        list: List of text segments that are within the token limit.
    """
    # Initialize the queue and result list
    queue = deque(texts)
    result = []

    while queue:
        # Pop the next segment from the queue
        segment = queue.popleft()

        # Skip empty or whitespace-only segments
        if not segment.strip():
            continue

        # Compute token count once per segment
        token_count = get_token_count(segment)
        if token_count == 0:
            continue

        # If the segment exceeds max size, attempt semantic splitting
        if token_count > max_chunk_size:
            sub_segments = semantic_splitter.split_text(segment)

            # If no valid sub-segments were created, fallback to hard split
            if not sub_segments or sub_segments == [segment]:
                fallback_chunks = hard_split(segment, max_chunk_size, overlap=int(max_chunk_size * 0.1))
                for fc in fallback_chunks:
                    result.append(fc)
                continue

            # Reinsert sub-segments at the front of the queue in reverse order
            for sub in reversed(sub_segments):
                queue.appendleft(sub)
            continue

        # If the segment is within the token limit, add it to the result
        result.append(segment)

    return result

def semantic_chunk_text_file(
    file_path: Path,
    embed_model_name: str,
    max_chunk_size: int,
    buffer_size: int,
    breakpoint_threshold_type: str,
    breakpoint_threshold_amount: float,
    sentence_split_regex: str,
    verbose: bool,
):
    """Processes a single text file and returns semantic chunks.

    Args:
        file_path (Path): Path to the text file to be processed.
        embed_model_name (str): Name of the embedding model to use.
        max_chunk_size (int): Maximum allowed size for each chunk in tokens.
        buffer_size (int): Buffer size for the semantic chunker.
        breakpoint_threshold_type (str): Type of breakpoint threshold for chunking.
        breakpoint_threshold_amount (float): Amount for the breakpoint threshold.
        sentence_split_regex (str): Regular expression for splitting sentences.
        verbose (bool): If True, prints detailed processing information.

    Returns:
        list: List of semantic chunks extracted from the file.
    """
    # Initialize embeddings and splitters
    embedding_model = OllamaEmbeddings(
        model=embed_model_name,             # The Ollama model name
        base_url=OLLAMA_URL   # Default Ollama server address: "http://localhost:11434"
    )

    semantic_splitter = SemanticChunker(
        embedding_model,
        buffer_size=buffer_size,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
        sentence_split_regex=sentence_split_regex,
    )

    # headers_to_split_on = [
    #     ("#", "header_1"),
    #     ("##", "header_2"),
    #     ("###", "header_3"),
    # ]
    # markdown_splitter = MarkdownHeaderTextSplitter(
    #     headers_to_split_on=headers_to_split_on,
    #     strip_headers=False
    # )

    with open(file_path, "r", encoding="utf-8") as f:
        input_text = f.read()

    if verbose:
        print(f"\nReading file: {file_path}")

    # Markdown split
    # md_header_splits = markdown_splitter.split_text(input_text)
    # plain_content = [value.page_content for value in md_header_splits]
    # semantic_chunks = process_semantic_split(plain_content, semantic_splitter, max_chunk_size)


    # Semantic split
    semantic_chunks = process_semantic_split(
        [input_text], semantic_splitter, max_chunk_size
    )

    return semantic_chunks

def chunk_text_data(
    documents_dir: str,
    embed_model_name: str,
    parent_chunk_size: int,
    child_chunk_size: int,
    documents_chunks_dir: str,
    buffer_size: int,
    breakpoint_threshold_type: str,
    breakpoint_threshold_amount: float,
    sentence_split_regex: str,
    verbose: bool,
):
    """
    Performs txt documents chunking using semantic chunking. Then, stores the resulting chunks into a dedicated folder.

    Args:
        documents_dir (str): Path to the directory containing text files to be chunked.
        embed_model_name (Dict): Dictionary containing the embedding model name and parameters.
        parent_chunk_size (int): Maximum allowed size for each parent chunk in tokens.
        child_chunk_size (int): Maximum allowed size for each child chunk in tokens.
        documents_chunks_dir (str): Path to the directory where chunks will be saved.
        buffer_size (int): Buffer size for the semantic chunker.
        breakpoint_threshold_type (str): Type of breakpoint threshold for chunking.
        breakpoint_threshold_amount (float): Amount for the breakpoint threshold.
        sentence_split_regex (str): Regular expression for splitting sentences.
        verbose (bool): If True, prints detailed processing information.

    Returns:
        None: The function saves the chunks to the specified directory.
    """
    documents_path = Path(documents_dir)
    documents_chunks_path = Path(documents_chunks_dir)

    documents_path.mkdir(parents=True, exist_ok=True)
    documents_chunks_path.mkdir(parents=True, exist_ok=True)

    # If child_chunk_size is not provided or is not less than parent_chunk_size,
    # set a default child_chunk_size to be 1/8 of the parent_chunk_size
    if child_chunk_size is None or not child_chunk_size < parent_chunk_size:
        child_chunk_size = parent_chunk_size // 8

    if verbose:
        print(f"\nProcessing the files in {documents_path}...\n")

    # In case of an original text file
    for file_path in documents_path.rglob("*_Text.md"):
        if file_path.is_file():
            # Process each text file and create semantic chunks (parent chunks)
            parent_chunks = semantic_chunk_text_file(
                file_path,
                embed_model_name,
                parent_chunk_size,
                buffer_size,
                breakpoint_threshold_type,
                breakpoint_threshold_amount,
                sentence_split_regex,
                verbose,
            )
            for idx, chunk in enumerate(parent_chunks, 1):
                chunk_file_name = (
                    documents_chunks_path / f"{file_path.stem}_parent_chunk_{idx}.txt"
                )  # documents_chunks_path / f"{file_path.stem}_chunk_{idx}.md"
                with open(chunk_file_name, "w", encoding="utf-8") as file:
                    file.write(chunk)
                if verbose:
                    print(f"Saved chunk to: {chunk_file_name}")

                # Create child chunks
                child_chunks = hard_split(chunk, child_chunk_size)  # Adjust size for child chunks
                for child_idx, child_chunk in enumerate(child_chunks, 1):
                    child_chunk_file_name = (
                        documents_chunks_path
                        / f"{file_path.stem}_parent_chunk_{idx}_child_chunk_{child_idx}.txt"
                    )
                    with open(child_chunk_file_name, "w", encoding="utf-8") as file:
                        file.write(child_chunk)
                    if verbose:
                        print(f"Saved child chunk to: {child_chunk_file_name}")

    # In case of an image description file
    for file_path in documents_path.rglob("*_description.txt"):
        if file_path.is_file():
            # Create only child chunks
            # The full description remains as a single parent chunk
            with open(file_path, "r", encoding="utf-8") as f:
                description = f.read()
            # Clean the description to remove table and code blocks, marmidown headers, etc.
            description = clean_up_text(description)
            chunks = hard_split(description, parent_chunk_size)  # Adjust size for child chunks
            for idx, chunk in enumerate(chunks, 1):
                chunk_file_name = (
                    documents_chunks_path
                    / f"{file_path.stem}_chunk_{idx}.txt"
                )
                with open(chunk_file_name, "w", encoding="utf-8") as file:
                    file.write(chunk)
                if verbose:
                    print(f"Saved child chunk to: {chunk_file_name}")

    if verbose:
        print(
            f"\nAll files have been processed.\nData chunks were saved in TXT to: {documents_chunks_dir}.\n"
        )


