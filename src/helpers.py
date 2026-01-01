import os
import re
import torch
import tiktoken
from PIL import Image, ImageStat
from pathlib import Path
from typing import List, Dict

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    SimpleKeywordTableIndex
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.llms import ChatMessage

from llama_index.core.indices.base import BaseIndex

import ollama
import json

import warnings
from typing import List, Tuple


from httpx import ReadTimeout
import markdown
from bs4 import BeautifulSoup

from .constants import (
    MEMORY_WINDOW,
    SYSTEM_PROMPT_REWRITE_QUERY,
    USER_PROMPT_REWRITE_QUERY,
    SYSTEM_PROMPT_EXPAND_QUERY,
    USER_PROMPT_EXPAND_QUERY
)


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")



def clear_pytorch_cache():
    """
    Clears MPS cache in PyTorch if available.
    """
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("\nCache cleared. Now running the rest of the code.\n")


def load_prompt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def is_relevant_image(
        image_path: Path,
        image_pixel_threshold: int,
        image_pixel_variance_threshold: float
):
    """
    Determines if an image is relevant based on non-white pixel count and color variance.

    Parameters:
    - image_path (Path): Path to the image file.
    - pixel_threshold (int): Minimum number of non-white pixels required.
    - variance_threshold (float): Minimum average variance across color channels.

    Returns:
    - bool: True if the image is relevant, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            # Calculate non-white pixels in grayscale
            grayscale = img.convert("L")
            histogram = grayscale.histogram()
            non_white = sum(histogram[:-1])

            if non_white <= image_pixel_threshold:
                return False

            # Calculate average variance across all color channels
            stat = ImageStat.Stat(img)
            avg_variance = sum(stat.var) / len(stat.var)

            if avg_variance < image_pixel_variance_threshold:
                return False

            return True
    except (IOError, OSError) as e:
        print(f"Error processing {image_path}: {e}")
        return False

def markdown_to_plain_text(markdown_text: str) -> str:
    """
    Converts Markdown text to plain text by first converting it to HTML
    and then stripping HTML tags.

    Args:
        markdown_text (str): The Markdown-formatted text.

    Returns:
        str: The plain text representation.
    """
    # Convert Markdown to HTML
    html = markdown.markdown(markdown_text)

    # Use BeautifulSoup to parse HTML and extract text
    soup = BeautifulSoup(html, "html.parser")
    plain_text = soup.get_text()

    return plain_text


def ask_LLM_with_JSON(
        prompt: str,
        ollama_client: ollama.Client,
        model: str,
        model_options: Dict[str, str]
):
    """
    Asks the LLM to generate a response based on the provided prompt.
    The response is expected to be in JSON format.
    Args:
        - prompt (str): The prompt to send to the LLM.
        - ollama_client (ollama_client): The Ollama client to use for querying the model.
        - model (str): The name of the model to use for querying.
        - model_options (Dict[str, str]): A dictionary containing options for the model, such as temperature.

    Returns:
        - str: The response from the LLM, expected to be in JSON format.
    """ 

    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant, who helps the user with their query. You are designed to output JSON."})     
    messages.append({"role": "user", "content": prompt})   

    response = ollama_client.chat(
        model=model,
        messages=messages,
        options=model_options
    )

    return response["message"]["content"]

def recover_json(json_str, verbose = False):
    """"
    Attempts to recover a JSON object from a string that may be malformed or improperly formatted.

    Args:
        - json_str (str): The string containing the JSON data, which may be malformed.
        - verbose (bool): If True, prints detailed information about the recovery process.

    Returns:
        - dict: The recovered JSON object if successful, or the original string if recovery fails.
    """
    decoded_object = {}

    if '{' not in json_str:
        return json_str

    json_str = extract_json(json_str)

    try:
        decoded_object = json.loads(json_str)
        return decoded_object
    except Exception:
        try:
            decoded_object = json.loads(json_str.replace("'", '"'))
            return decoded_object
        except Exception:
            try:
                decoded_object = json_repair.loads(json_str.replace("'", '"'))

                for k, d in decoded_object.items():
                    dd = d.replace("'", '"')
                    decoded_object[k] = json.loads(dd)
                
                return decoded_object
            except:
                print(f"all json recovery operations have failed for {json_str}")
        
            if verbose:
                if isinstance(decoded_object, dict):
                    print(f"\n{bc.OKBLUE}>>> Recovering JSON:\n{bc.OKGREEN}{json.dumps(decoded_object, indent=3)}{bc.ENDC}")
                else:
                    print(f"\n{bc.OKBLUE}>>> Recovering JSON:\n{bc.OKGREEN}{json_str}{bc.ENDC}")


    return json_str

def extract_json(s: str) -> str:
    """
    Extracts a JSON string from a larger string that may contain Markdown formatting.
    This function looks for a code block in the format ```json ... ``` and returns the content inside it.
    If no such code block is found, it returns the original string.
    Args:
        - s (str): The input string that may contain a JSON code block.
    Returns:
        - str: The extracted JSON string if found, otherwise the original string.
    """
    # Use regex to find a code block with language 'json'
    code = re.search(r"```json(.*?)```", s, re.DOTALL)
    if code:
        return code.group(1)
    else:
        return s
    
def extract_markdown_tables(text: str) -> list[str]:
    """
    Finds all Markdown tables in the text.
    A Markdown table is defined as:
      1) A header row    (a line starting and ending with '|')
      2) A separator row (a line of pipes, dashes, spaces, or colons)
      3) One or more data rows (lines starting and ending with '|')

    The tables are expected to be in the format:

    | Column1 | Column2 |
    |---------|---------|
    | Data1   | Data2   |

    If no tables are found, it returns an empty list.

    Returns a list of the raw table strings.
    """
    table_pattern = (
        r'(?:^[ \t]*\|.*\r?\n)'                    # header row
        r'(?:^[ \t]*\|[-\s|:]+\r?\n)'              # separator row
        r'(?:^[ \t]*\|.*\r?\n?)+'                  # one or more data rows
    )
    return re.findall(table_pattern, text, flags=re.MULTILINE)

def remove_markdown_tables(text: str) -> str:
    """
    Removes all Markdown tables (as defined above) from the text.
    """
    table_pattern = (
        r'(?:^[ \t]*\|.*\r?\n)'
        r'(?:^[ \t]*\|[-\s|:]+\r?\n)'
        r'(?:^[ \t]*\|.*\r?\n?)+'
    )
    return re.sub(table_pattern, '', text, flags=re.MULTILINE)

def remove_code_blocks(text: str) -> str:
    """
    Strips out any fenced code block (```...```), regardless of language.
    """
    return re.sub(r'```.*?```', '', text, flags=re.DOTALL)

def remove_mermaid_blocks(text: str) -> str:
    """
    Specifically strips out any ```mermaid ... ``` block.
    """
    return re.sub(r'```mermaid.*?```', '', text, flags=re.DOTALL)

def remove_extracted_text_blocks(text: str) -> str:
    """
    Removes any ```EXTRACTED TEXT ... ``` sections.
    """
    return re.sub(r'```EXTRACTED TEXT.*?```', '', text, flags=re.DOTALL)

def clean_up_text(text: str) -> str:
    """
    Cleans up the input text by:
      1. Removing all fenced code blocks
      2. Removing mermaid diagrams
      3. Removing EXTRACTED TEXT sections
      4. Removing all Markdown tables
    """
    text = remove_code_blocks(text)
    text = remove_mermaid_blocks(text)
    # text = remove_extracted_text_blocks(text)
    text = remove_markdown_tables(text)
    return text

def get_token_count(input_text: str) -> int:
    """Returns the number of tokens for the input text."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(input_text))

def load_embeddings_with_associated_documents(embeddings_dir: str, chunks_dir: str, docs_dir: str) -> List[Dict]:
    """
    Loads the embeddings and the associated source documents (children and parents) from corresponding directories.

    Args:
        - embeddings_dir (str): The directory containing the embedding files with their associated metadada.
        - chunks_dir (str): The directory containing the associated data chunks.
        - docs_dir (str): The directory containing the associated parent documents.

    Returns:
        A list containing all the texts with associated embeddings.
    """
    embeddings_path = Path(embeddings_dir)
    chunks_path = Path(chunks_dir)
    docs_path = Path(docs_dir)
   
    embeddings, child_docs, parent_docs = [], [], []

    print(f"Loading embeddings from directory: {embeddings_path} ...\n")

    for embedding_file in embeddings_path.rglob("*.json"):
        print(f"\nLoading the embedding file : {embedding_file.name}...")
        with open(embedding_file, "r", encoding="utf-8") as f:
            embedding_json = json.load(f)

        # Get the embedding vector    
        embeddings.append(embedding_json["embedding"])

        # Get the child document
        child_doc_path = chunks_path / f"{embedding_json["metadata"]["child"]}"
        with open(child_doc_path, "r", encoding="utf-8") as f:
            child_doc_text = f.read()
        child_doc = {
            "text": child_doc_text,
            "metadata": {
                "name": embedding_json["metadata"]["child"],
                "parent": embedding_json["metadata"]["parent"],
                "source": embedding_json["metadata"]["source"]
            }
        }
        child_docs.append(child_doc)

        # Get the parent document
        parent_doc_name = embedding_json["metadata"]["parent"]
        if "_parent_chunk" in parent_doc_name:
            # If the parent document is a chunk, we need to load it from the chunks directory
            parent_doc_path = chunks_path / f"{parent_doc_name}"
        else:
            # If the parent document is a full document, we load it from the docs directory
            parent_doc_path = docs_path / f"{parent_doc_name}"

        with open(parent_doc_path, "r", encoding="utf-8") as f:
            parent_doc_text = f.read()
        parent_doc = {
            "text": parent_doc_text,
            "metadata": {
                "name": parent_doc_name,
                "source": embedding_json["metadata"]["source"]
            }
        }
        parent_docs.append(parent_doc)

    print(f"""\n
          All embeddings were loaded from directory: {embeddings_path}.
          Associated child documents were loaded from: {chunks_path}.
          And associated parent documents from: {chunks_path}
        \n""")

    return embeddings, child_docs, parent_docs



# The following functions are used to buil the RAG app.


# Helper function to create Document instances with metadata
def create_documents(
    texts: List[str], metadata_list: List[Dict[str, str]]
) -> List[Document]:
    if len(texts) != len(metadata_list):
        raise ValueError(
            "The number of texts must match the number of metadata entries."
        )
    return [Document(text=t, metadata=m) for t, m in zip(texts, metadata_list)]


# Function to create an index for a given set of documents and storage context
# Will be used to create: child documents index for vector search, and parent ones for keyword-based search

def create_index(
    documents: List[Document],
    storage_context: StorageContext,
    index_name: str,
    index_type: str
):
    """
    Create an index from a given set of documents and a storage context.

    This function supports two index types:
      - "vector": Creates a VectorStoreIndex for semantic similarity searches.
      - "keyword": Creates a SimpleKeywordTableIndex for keyword-based searches.

    Args:
        documents (List[Document]): 
            A list of Document objects to be indexed.
        storage_context (StorageContext): 
            The storage context for saving/retrieving index data.
        index_name (str): 
            A descriptive name for the index (used in logging).
        index_type (str):
            The type of index to create. Must be either "vector" or "keyword".

    Returns:
        BaseIndex: The newly created index (VectorStoreIndex or SimpleKeywordTableIndex).

    Raises:
        ValueError: If an unsupported index_type is given.
        IOError or OSError: If index creation fails due to I/O issues.
    """
    try:
        logger.info("Creating %s...", index_name)

        # Branch based on the chosen index type
        if index_type == "vector":
            # Create a vector-based index for semantic similarity
            index = VectorStoreIndex.from_documents(
                documents=documents, 
                storage_context=storage_context
            )
        elif index_type == "keyword":
            # Create a keyword-based index for simple keyword lookup
            index = SimpleKeywordTableIndex.from_documents(
                documents=documents, 
                storage_context=storage_context
            )
        else:
            # Raise a ValueError for clarity when invalid index_type is used
            raise ValueError("Argument 'index_type' must be either 'vector' or 'keyword'.")

        logger.info("Index '%s' created successfully.", index_name)
        return index

    except (IOError, OSError) as e:
        logger.error("Failed to create index '%s': %s", index_name, e)
        raise

# Function to save index to disk for faster loading and for not recreating it from scratch.
def save_index(index: BaseIndex, save_dir: Path):
    """
    Saves the given index to the specified directory.

    Args:
        index (BaseIndex): The index to save, which can be a VectorStoreIndex or SimpleKeywordTableIndex.
        save_dir (Path): The directory where the index will be saved.

    Raises:
        IOError or OSError: If the index cannot be saved due to I/O issues.
    """
    os.makedirs(save_dir, exist_ok=True)
    try:
        index.storage_context.persist(persist_dir=save_dir)
        logger.info("Index saved to %s.", save_dir)
    except (IOError, OSError) as e:
        logger.error("Failed to save index to %s: %s", save_dir, e)
        raise


# Function for querying an index
def query_index(index: VectorStoreIndex, query: str):
    """
    Queries the given index with the provided query string.

    Args:
        index (VectorStoreIndex): The index to query.
        query (str): The query string to search for.

    Returns:
        The response from the index query.

    Raises:
        IOError or OSError: If the query fails due to I/O issues.
    """
    try:
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        print(f"Response from {index}: {response}")
        return response
    except (IOError, OSError) as e:
        logger.error("Failed to query index: %s", e)


# Performs metadata-based search from an index
# Will be used to retrieve parent documents from child ones previously retrieved by similarity serach
def retrieve_from_keyword_index(index, key, value):
    """Return a list of Documents whose metadata[key] == value."""
    docstore = index.docstore
    results = []
    for doc_id, doc_obj in docstore.docs.items():
        if doc_obj.metadata.get(key) == value:
            results.append(doc_obj)
    return results

def limit_chat_history(history: List[dict], max_tokens: int = MEMORY_WINDOW) -> List[dict]:
    """
    Truncates the chat history to fit within a maximum token limit.

    Args:
        history (List[dict]): The chat history, where each dict contains 'role' and 'content'.
        max_tokens (int): The maximum number of tokens allowed in the history.

    Returns:
        List[dict]: The truncated chat history that fits within the token limit.
    """
    total_tokens = 0
    truncated = []
    for msg in reversed(history):
        tokens = get_token_count(msg["content"])
        if total_tokens + tokens > max_tokens:
            break
        truncated.append(msg)
        total_tokens += tokens
    return list(reversed(truncated))


# 2) Streaming utilities
def stream_llm_output(llm: Ollama, messages: List[ChatMessage]):
    """
    Streams the output of an LLM chat response.

    Args:
        llm (Ollama): The LLM instance to use for streaming.
        messages (List[ChatMessage]): The chat messages to send to the LLM.

    Yields:
        str: The partial response from the LLM as it streams.
    """
    for token in llm.stream_chat(messages):
        yield token.delta


def run_streaming(llm: Ollama, messages: List[ChatMessage]):
    """
    Runs the LLM in streaming mode and yields the response as it is generated.

    Args:
        llm (Ollama): The LLM instance to use for streaming.
        messages (List[ChatMessage]): The chat messages to send to the LLM.

    Yields:
        str: The accumulated response from the LLM as it streams.
    """
    response = ""
    for partial in stream_llm_output(llm, messages):
        response += partial
        yield response


# Rewrite query
def rewrite_query(ollama_llm: Ollama, user_query: str, conversation_history_str: str = "") -> str:
    """
    Rewrites the user query based on the conversation history and the new question.

    Args:
        ollama_llm (Ollama): The LLM instance to use for rewriting.
        user_query (str): The new question to rewrite.
        conversation_history_str (str): The conversation history as a string.

    Returns:
        str: The rewritten question, formatted as a bullet point if multiple questions are present.
    """
    with open(SYSTEM_PROMPT_REWRITE_QUERY, "r", encoding="utf-8") as s_prompt:
        system_message = s_prompt.read()

    with open(USER_PROMPT_REWRITE_QUERY, "r", encoding="utf-8") as u_prompt:
        user_message = u_prompt.read()

    user_prompt = user_message.format(
        conversation_history=conversation_history_str,
        question=user_query
    )
    messages = [
        ChatMessage(role="system", content=system_message),
        ChatMessage(role="user", content=user_prompt),
    ]
    response = ollama_llm.chat(messages)
    rewritten = response.message.content.strip()
    return rewritten or f"- {user_query}"

def parse_rewritten_query(text: str) -> List[str]:
    """
    Extracts bullet-pointed questions from the rewritten query.

    Each line starting with '- ' is treated as a separate item.
    Returns a list of the lines with the '- ' prefix removed.

    Args:
        - text (str): The rewritten query as a string

    Returns:
        List[str]: The list of the sub-queries rewritten
    """
    lines = text.splitlines()
    bullets: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- '):
            # remove the leading '- ' and any extra whitespace
            bullets.append(stripped[2:].strip())
    return bullets

# Query expansion with timeout and fallback
def expand_query(ollama_llm: Ollama, user_query: str) -> List[str]:
    """
    Expands the user query by identifying synonyms or related terms using the LLM.

    Args:
        ollama_llm (Ollama): The LLM instance to use for query expansion.
        user_query (str): The user query to expand.

    Returns:
        List[str]: A list of expanded queries, or the original query if expansion fails.
    """
    try:
        sub_questions = [l.strip("-* \t") for l in user_query.splitlines() if l.strip()]
        if not sub_questions:
            sub_questions = [user_query]

        all_expanded = []
        with open(SYSTEM_PROMPT_EXPAND_QUERY, "r", encoding="utf-8") as s_prompt:
            system_message = s_prompt.read()
        with open(USER_PROMPT_EXPAND_QUERY, "r", encoding="utf-8") as u_prompt:
            user_message = u_prompt.read()

        for sq in sub_questions:
            user_prompt = user_message.format(
                subq=sq
            )
            messages = [
                ChatMessage(role="system", content=system_message),
                ChatMessage(role="user", content=user_prompt),
            ]
            resp = ollama_llm.chat(messages, stream=False)
            cands = parse_expanded_queries(resp.message.content)
            all_expanded.extend(cands or [sq])
        return all_expanded
    except ReadTimeout:
        logger.warning("Query expansion timed out – using raw query")
        return [user_query]
    except Exception as e:
        logger.error("Error during query expansion: %s", e)
        return [user_query]

# Parse expansions
def parse_expanded_queries(response_text: str, max_queries: int = 5) -> List[str]:
    """
    Parses the expanded queries from the LLM response text.

    Args:
        response_text (str): The text response from the LLM containing expanded queries.
        max_queries (int): The maximum number of queries to return.

    Returns:
        List[str]: A list of expanded queries extracted from the response text.
    """
    marker = "Step 3"
    start = response_text.find(marker)
    if start == -1:
        return []
    lines = response_text[start:].splitlines()
    queries = []
    capture = False
    for line in lines:
        if "Expanded Queries:" in line:
            capture = True
            continue
        if capture and line.strip().startswith("-"):
            queries.append(line.strip().lstrip("- "))
    return queries[:max_queries]

# Convert to Gradio tuples
def convert_history_to_tuples(history: List[dict]) -> List[Tuple[str, str]]:
    """
    Converts a chat history list of dictionaries into a list of tuples for Gradio.
    Each tuple contains the user message and the assistant response.

    Args:
        history (List[dict]): The chat history, where each dict contains 'role' and 'content'.

    Returns:
        List[Tuple[str, str]]: A list of tuples, each containing a user message and an assistant response.
    """
    out = []
    user_msg = None
    for msg in history:
        if msg["role"] == "user":
            user_msg = msg["content"]
        elif msg["role"] == "assistant":
            out.append((user_msg or "", msg["content"]))
            user_msg = None
    if user_msg is not None:
        out.append((user_msg, ""))
    return out


# Reset Conversation
def reset_conversation():
    """
    Resets the conversation history and returns an empty list and an empty string.
    Returns:
        Tuple[List[dict], str]: An empty list for the conversation history and an empty string for the current query.
    """
    return [], ""