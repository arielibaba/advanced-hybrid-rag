# app.py

import os
from pathlib import Path
import logging
import nest_asyncio
import gradio as gr
import multiprocessing
import warnings

from dotenv import load_dotenv, find_dotenv

from .constants import (
    INDEX_DIR,
    CHILD_DOCUMENTS_INDEX,
    PARENT_DOCUMENTS_INDEX,
    TOP_K_RETRIEVED_CHILDREN,
    TOP_K_RERANKED_PARENTS,
    TOP_K_REFS,
    LLM,
    EMBED_MODEL,
    TEMPERATURE_GENERATION,
    TOP_P_GENERATION,
    OLLAMA_URL,
    OLLAMA_REQUEST_TIMEOUT,
    MEMORY_WINDOW,
    SYSTEM_PROMPT_GENERATE_FINAL_ANSWER,
    USER_PROMPT_GENERATE_FINAL_ANSWER
)
from .helpers import (
    clear_pytorch_cache,
    retrieve_from_keyword_index,
    limit_chat_history,
    run_streaming,
    rewrite_query,
    parse_rewritten_query,
    expand_query,
    convert_history_to_tuples,
    reset_conversation,
)

from llama_index.core import  (
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.llms import ChatMessage

import ollama

import time  # ⬅️ Added for profiling

# Suppress and async patch
warnings.filterwarnings("ignore")
nest_asyncio.apply()

# Load env & logging
load_dotenv(find_dotenv())
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fh = logging.FileHandler("diagnostics_response_time.log")
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(fh)
logger.info("Using Ollama request timeout = %s", OLLAMA_REQUEST_TIMEOUT)

# Clear the cache
clear_pytorch_cache()

# Ollama clients
NUM_CPUS = multiprocessing.cpu_count()
ollama_client = ollama.Client()

ollama_llm = Ollama(
    model=LLM,
    base_url=OLLAMA_URL,
    temperature=TEMPERATURE_GENERATION,
    additional_kwargs={"top_p": TOP_P_GENERATION},
    request_timeout=OLLAMA_REQUEST_TIMEOUT,
)
ollama_embed = OllamaEmbedding(
    model_name=EMBED_MODEL,
    base_url=OLLAMA_URL,
    ollama_additional_kwargs={},
)

Settings.llm = ollama_llm
Settings.embed_model = ollama_embed

# Load indices
def load_index(name: str):
    path = Path(INDEX_DIR) / name
    ctx = StorageContext.from_defaults(persist_dir=path)
    return load_index_from_storage(ctx)

child_index = load_index(CHILD_DOCUMENTS_INDEX)
parent_index = load_index(PARENT_DOCUMENTS_INDEX)

# Reranker & retriever
reranker = LLMRerank(choice_batch_size=5, top_n=TOP_K_RERANKED_PARENTS)
child_retriever = VectorIndexRetriever(index=child_index, similarity_top_k=TOP_K_RETRIEVED_CHILDREN)


def chat_conversation(user_message, history):
    t0 = time.perf_counter()

    # normalize history
    if history is None:
        history = []
    elif history and isinstance(history[0], (list, tuple)):
        flat = []
        for u, a in history:
            flat += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        history = flat

    history.append({"role": "user", "content": user_message})
    history = limit_chat_history(history, max_tokens=MEMORY_WINDOW)
    conv_history = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in history)

    try:
        t1 = time.perf_counter()
        rewritten = rewrite_query(ollama_llm, user_message, conv_history)
        t2 = time.perf_counter()
        logger.info("Rewritten query (%d):\n%s", len(rewritten.split('\n')), rewritten)
    except Exception as e:
        logger.error("OLLAMA client timed out during query rewrite: %s", e)
        history.append({"role": "assistant", "content": "My apologies, the service timed out. Please try again later."})
        yield convert_history_to_tuples(history)
        return

    candidates = parse_rewritten_query(rewritten)

    # Retrieve & rerank
    t3 = time.perf_counter()
    retrieved = sum((child_retriever.retrieve(q) for q in candidates), [])
    parents = []
    for node in retrieved:
        pk = node.metadata.get("parent")
        p = retrieve_from_keyword_index(parent_index, "name", pk)
        if p:
            parents.append(p[0])

    # dedupe & rerank
    seen = set(); unique = []
    for p in parents:
        name = p.metadata.get("name")
        if name not in seen:
            seen.add(name); unique.append(p)
    nws_list = [NodeWithScore(node=p, score=0.0) for p in unique]
    combo = user_message + " | " + " | ".join(candidates)
    reranked = reranker.postprocess_nodes(nws_list, query_str=combo)
    t4 = time.perf_counter()

    # build context & references
    context = "\n".join(n.node.text for n in reranked)
    refs = []
    seen_pages = set()
    for i, nws in enumerate(reranked, 1):
        doc = nws.node.metadata["source"]["document"]
        page = nws.node.metadata["source"]["page"]
        if (doc, page) not in seen_pages:
            seen_pages.add((doc, page))
            refs.append(f"{i}. {doc} - Page {page}")
        if len(refs) >= TOP_K_REFS:
            break

    # LLM prompt
    with open(SYSTEM_PROMPT_GENERATE_FINAL_ANSWER, "r", encoding="utf-8") as s_prompt:
        system_msg = s_prompt.read()

    with open(USER_PROMPT_GENERATE_FINAL_ANSWER, "r", encoding="utf-8") as u_prompt:
        user_msg_template = u_prompt.read()

    user_prompt = user_msg_template.format(
        conversation_history=conv_history,
        user_question=user_message,
        refined_context=context
    )
    msgs = [ChatMessage(role="system", content=system_msg),
            ChatMessage(role="user", content=user_prompt)]

    t5 = time.perf_counter()

    history.append({"role": "assistant", "content": ""})
    for chunk in run_streaming(ollama_llm, msgs):
        history[-1]["content"] = chunk
        yield convert_history_to_tuples(history)

    t6 = time.perf_counter()

    final = history[-1]["content"].strip()
    if not final.lower().startswith("no relevant details"):
        final += "\n\n---\nReferences:\n" + "\n".join(refs)
    history[-1]["content"] = final

    # ⏱️ Log performance
    logger.info(f"""
    ⏱️ Profiling Stats:
    - Query Rewriting:      {t2 - t1:.2f}s
    - Retrieval:            {t3 - t2:.2f}s
    - Reranking:            {t4 - t3:.2f}s
    - Prompt Construction:  {t5 - t4:.2f}s
    - LLM Inference:        {t6 - t5:.2f}s
    - TOTAL:                {t6 - t0:.2f}s
    """)

    yield convert_history_to_tuples(history)


# Gradio app
with gr.Blocks() as demo:
    gr.Markdown("# WatchComplyChat\n\nA Gen AI-Powered Legal Analytics Solution that Analyzes All the Sanctions from French and European Regulators.")
    chatbot = gr.Chatbot(height=600, label="WatchComplyChat")
    user_input = gr.Textbox(label="Your message", placeholder="Type here…", lines=1)
    submit_btn = gr.Button("Submit")
    reset_btn  = gr.Button("Reset Conversation")

    submit_btn.click(chat_conversation, [user_input, chatbot], chatbot, queue=True)
    user_input.submit(chat_conversation, [user_input, chatbot], chatbot, queue=True)
    reset_btn.click(reset_conversation, [], [chatbot, user_input], queue=False)

if __name__ == "__main__":
    logger.info("Launching the Chatbot Web App …")
    demo.launch()