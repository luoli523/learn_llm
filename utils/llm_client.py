"""
Unified LLM client for all notebooks.

Configure via .env — see .env.example for all supported providers.

Usage:
    from utils.llm_client import chat, stream_chat, multi_turn_chat

    response = chat("Explain what a token is.")
    for chunk in stream_chat("Write a haiku."):
        print(chunk, end="", flush=True)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
import litellm

# Load .env from repo root regardless of where the notebook is located
_root = Path(__file__).resolve().parents[1]
load_dotenv(_root / ".env")

# Suppress provider-specific params that a given model doesn't support
litellm.drop_params = True


def _model() -> str:
    model = os.getenv("LLM_MODEL")
    if not model:
        raise EnvironmentError(
            "LLM_MODEL is not set. Copy .env.example to .env and configure it."
        )
    return model


def chat(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs,
) -> str:
    """Single-turn chat. Returns the full response as a string."""
    messages = _build_messages(system, prompt)
    response = litellm.completion(
        model=_model(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return response.choices[0].message.content or ""


def stream_chat(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs,
) -> Generator[str, None, None]:
    """Single-turn streaming chat. Yields text chunks as they arrive."""
    messages = _build_messages(system, prompt)
    response = litellm.completion(
        model=_model(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        **kwargs,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def multi_turn_chat(
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs,
) -> str:
    """
    Multi-turn chat. Pass the full message history.

    Example:
        messages = [
            {"role": "system", "content": "You are a helpful teacher."},
            {"role": "user", "content": "What is a token?"},
            {"role": "assistant", "content": "A token is ..."},
            {"role": "user", "content": "How is that different from a word?"},
        ]
        reply = multi_turn_chat(messages)
    """
    response = litellm.completion(
        model=_model(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return response.choices[0].message.content or ""


def _build_messages(system: str | None, prompt: str) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages
