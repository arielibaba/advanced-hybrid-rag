"""Async utilities for running coroutines from mixed sync/async contexts."""

import asyncio
import concurrent.futures
from typing import Any, Coroutine


def run_coroutine(coro: Coroutine) -> Any:
    """
    Run an async coroutine from any context (sync or async).

    If an event loop is already running (e.g. inside Gradio, Jupyter, or an async
    pipeline), the coroutine is submitted to a single-threaded pool so it gets its
    own fresh event loop without nesting.

    If no loop is running, ``asyncio.run()`` is used directly.

    Args:
        coro: The coroutine to execute.

    Returns:
        The coroutine's return value.
    """
    try:
        asyncio.get_running_loop()
        # Already in an async context — run in a new thread to avoid nesting
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No event loop running — safe to use asyncio.run()
        return asyncio.run(coro)
