# src/models/llm_client.py

"""
Unified LLM client for:
- Judge calls (JSON-required)
- Refinement calls (free-form text)

Supports the new OpenAI API format:
    - max_completion_tokens instead of max_tokens
    - temperature only allowed for some models
"""

from dataclasses import dataclass
from typing import Any, Dict, List
import json
import os
from openai import OpenAI

# ======================================================
# Load .env from project root (bulletproof absolute path)
# ======================================================
from dotenv import load_dotenv

# Find project root: src/models/ → src/ → project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_PATH = os.path.join(ROOT_DIR, ".env")

load_dotenv(ENV_PATH)  # loads OPENAI_API_KEY correctly no matter how script is run


# ======================================================
# Configuration Dataclass
# ======================================================

@dataclass
class LLMConfig:
    model: str
    max_completion_tokens: int = 512
    temperature: float | None = None  # Only used when model allows it
    seed: int | None = None


# ======================================================
# Core LLM Call
# ======================================================

def call_llm(
    system_prompt: str,
    user_prompt: str,
    cfg: LLMConfig,
    json_mode: bool = False
) -> str:
    """
    Execute a chat completion with the given model configuration.

    Args:
        system_prompt: system role message
        user_prompt: user role message
        cfg: LLMConfig object
        json_mode: if True → force JSON output using 'response_format'

    Returns:
        The model's response text.
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Build request payload
    request: Dict[str, Any] = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_completion_tokens": cfg.max_completion_tokens,
    }

    # Temperature allowed only for certain models
    if cfg.temperature is not None:
        request["temperature"] = cfg.temperature

    # Seeds supported on gpt-5-series
    if cfg.seed is not None:
        request["seed"] = cfg.seed

    # JSON mode enabled if requested
    if json_mode:
        request["response_format"] = {"type": "json_object"}

    # Send request
    response = client.chat.completions.create(**request)

    # Return the text output
    return response.choices[0].message.content


# ======================================================
# Robust JSON parser
# ======================================================

def parse_json_or_throw(text: str) -> Dict[str, Any]:
    """
    Attempts strict JSON loading.
    If it fails, searches for the first {...} block.
    """
    try:
        return json.loads(text)
    except Exception:
        # Fallback: extract JSON substring
        import re
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        raise ValueError(f"LLM did not return valid JSON.\nOutput:\n{text}")
