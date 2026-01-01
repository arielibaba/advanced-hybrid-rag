import os
import logging
from pathlib import Path

import ollama

from llama_index.core import (
    StorageContext,
    Settings,
    SimpleKeywordTableIndex
)

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.vector_stores.qdrant import QdrantVectorStore


from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse
)

from .helpers import (
    load_embeddings_with_associated_documents,
    create_documents,
    create_index,
    save_index
)
from .constants import (
    PROCESSED_DIR,
    CHUNKS_DIR,
    EMBEDDINGS_DIR,
    VECTOR_STORE_DIR,
    INDEX_DIR,
    CHILD_DOCUMENTS_INDEX,
    PARENT_DOCUMENTS_INDEX,
    LLM,
    EMBED_MODEL,
    TEMPERATURE_GENERATION,
    TOP_P_GENERATION,
    OLLAMA_URL,
    OLLAMA_REQUEST_TIMEOUT
)
from .create_embeddings import get_embeddings

import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize Ollama client
ollama_client = ollama.Client() 

# Models Configuration

# Generation Model 
# Ollama Thinking Model
ollama_llm = Ollama(
    model=LLM,
    base_url=OLLAMA_URL,
    temperature=TEMPERATURE_GENERATION,
    additional_kwargs={"top_p": TOP_P_GENERATION},
    request_timeout=OLLAMA_REQUEST_TIMEOUT,
)

# Embedding Model
ollama_embed = OllamaEmbedding(
    model_name=EMBED_MODEL,
    base_url=OLLAMA_URL,
    ollama_additional_kwargs={},   # e.g. {"mirostat": 0}
    client_kwargs=None,            # optional httpx.Client params
)

logger.info("initializing the global settings")
Settings.llm = ollama_llm
Settings.embed_model = ollama_embed


if __name__ == "__main__":

    for d in [VECTOR_STORE_DIR, INDEX_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Initialize Qdrant (only do this once at indexing time)
    print("\nInitialize Qdrant Client and Collections...\n")
    logger.info("Initialize Qdrant Client and Collections...")
    client = QdrantClient(path=VECTOR_STORE_DIR)

    # Determine embedding dimension using a simple test input
    test_input = "This is a test input to determine embedding dimensions."
    # OllamaEmbedding uses the method 'get_text_embedding' to get the embedding vector
    embed_vector = ollama_embed.get_text_embedding(test_input)
    embed_dim = len(embed_vector)
   
    # Define collection names
    collection_names = [
        "child_documents",
        "parent_documents"
    ]

    # Recreate collection only if needed
    # If data is static and unchanged, consider skipping recreation to speed up startup.
    for collection_name in collection_names:
        try:
            # Check if the collection already exists
            existing_collections = [c.name for c in client.get_collections().collections]
            if collection_name not in existing_collections:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embed_dim, distance=Distance.COSINE
                    ),
                )
                print(
                    f"Collection '{collection_name}' created with vector dimension {embed_dim}."
                )
                logger.info(
                    "Collection '%s' created with vector dimension %d.", collection_name, embed_dim
                )
            else:
                print(
                    f"Collection '{collection_name}' already exists; skipping recreation."
                )
                logger.info(
                    "Collection '%s' already exists; skipping recreation.", collection_name
                )
        except ResponseHandlingException as e:
            logger.error("Failed to create collection '%s' due to response handling error: %s", collection_name, e)
        except UnexpectedResponse as e:
            logger.error("Failed to create collection '%s' due to unexpected response: %s", collection_name, e)
        except (IOError, OSError) as e:
            logger.error("An unexpected error occurred while creating collection '%s': %s", collection_name, e)
            

    print("\nQdrant collections setup complete.")
    logger.info("Qdrant collections setup complete.")

    print("\nLoad the embeddings...\n")
    logger.info("Load the embeddings.")

    # Embebeddings
    embeddings, child_documents_with_metadata, parent_documents_with_metadata = load_embeddings_with_associated_documents(
        embeddings_dir=EMBEDDINGS_DIR,
        chunks_dir=CHUNKS_DIR,
        docs_dir=PROCESSED_DIR
    )

    # Initialize Qdrant Vector Stores
    children_vector_store = QdrantVectorStore(
        client=client,
        collection_name="child_documents"
    )
    parents_vector_store = QdrantVectorStore(
        client=client,
        collection_name="parent_documents"
    )

    # Define storage context
    children_storage_context = StorageContext.from_defaults()
    children_storage_context.vector_stores["children_vector_store"] = children_vector_store

    parents_storage_context = StorageContext.from_defaults()
    parents_storage_context.vector_stores["parents_vector_store"] = parents_vector_store

    logger.info("Prepare Documents with Metadata.")

    
    # Get the texts and the metadata for children and parents
    child_docs_texts = [child["text"] for child in child_documents_with_metadata]
    child_docs_metadata = [child["metadata"] for child in child_documents_with_metadata]

    parent_docs_texts = [parent["text"] for parent in parent_documents_with_metadata]
    parent_docs_metadata = [child["metadata"] for child in parent_documents_with_metadata]


    # Create the Child and Parent Document instances
    child_documents = create_documents(child_docs_texts, child_docs_metadata)
    parent_documents = create_documents(parent_docs_texts, parent_docs_metadata)
   
    print("\nCreate VectorStoreIndex Instance for Child Documents ...")
    logger.info("Create VectorStoreIndex Instances for Child Documents ...")

    # Building 2 Indexes for: Child and Parents Documents
    child_documents_index = create_index(
        documents=child_documents,
        storage_context=children_storage_context,
        index_name="child_documents",
        index_type="vector"
    )

    print("\nCreate KeywordTableIndex Instance for Parent Documents ...")
    logger.info("Create KeywordTableIndex Instances for Parent Documents ...")

    parent_documents_index = create_index(
        documents=parent_documents,
        storage_context=parents_storage_context,
        index_name="parent_documents",
        index_type="keyword"
    )

    print("\nSave the Indexes to Disk...\n")
    logger.info("Save the Indexes to Disk.")
    base_dir = Path(INDEX_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)
    # Save the indexes to disk
    save_index(child_documents_index, base_dir / CHILD_DOCUMENTS_INDEX)
    save_index(parent_documents_index, base_dir / PARENT_DOCUMENTS_INDEX)

    logger.info("The indexes have been saved to disk successfully to: %s.", base_dir)
    print(f"\nThe indexes have been saved to disk successfully to: {base_dir}.\n")
