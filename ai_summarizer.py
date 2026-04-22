"""
LangChain + Gemini squat summary agent.

Reads structured rep metrics and returns a concise coaching summary.
Set GEMINI_API_KEY in environment before enabling.
"""

from __future__ import annotations

import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


DEFAULT_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]


def _normalize_model_name(model_name: str) -> str:
    """
    Convert loose user style names into Gemini API-compatible format.
    Examples:
      "gemini 2.5 flash" -> "gemini-2.5-flash"
      "Gemini-1.5 Flash" -> "gemini-1.5-flash"
    """
    m = (model_name or "").strip().lower().replace("_", "-")
    while "  " in m:
        m = m.replace("  ", " ")
    m = m.replace(" ", "-")
    return m


def _build_prompt(payload: dict) -> str:
    return (
        "Analyse this squat rep metrics JSON and produce a short, practical summary.\n"
        "Output plain text with these sections:\n"
        "1) Overall quality (1-2 lines)\n"
        "2) Rep-by-rep highlights\n"
        "3) Top 3 corrections for next set\n"
        "4) Positive reinforcement (1 line)\n\n"
        "Rep metrics JSON:\n"
        f"{json.dumps(payload, indent=2)}"
    )


def summarize_squat_metrics(payload: dict) -> str:
    """
    Send summary request to Gemini via LangChain and return text.
    Returns user-friendly fallback text on config/network errors.
    """
    api_key = "AIzaSyDXoeA5r-eIpqzkWv2Aclk4QdjlZqBmGOo"
    if not api_key:
        return (
            "AI summary unavailable: GEMINI_API_KEY is not set.\n"
            "Set environment variable GEMINI_API_KEY and try again."
        )

    env_model = _normalize_model_name(os.getenv("GEMINI_MODEL", ""))
    model_candidates = [m for m in [env_model, *DEFAULT_MODEL_CANDIDATES] if m]
    seen = set()
    model_candidates = [m for m in model_candidates if not (m in seen or seen.add(m))]

    messages = [
        SystemMessage(
            content=(
                "You are an expert squat coach. Be concise, practical, and supportive. "
                "Use the provided metrics only; do not invent injuries or medical diagnoses."
            )
        ),
        HumanMessage(content=_build_prompt(payload)),
    ]

    last_error = ""
    for model_name in model_candidates:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.2,
            )
            response = llm.invoke(messages)
            text = getattr(response, "content", "")
            if isinstance(text, list):
                text = "".join(str(part) for part in text)
            text = str(text).strip()
            if text:
                return text
            last_error = f"empty response text from model '{model_name}'"
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            continue

    return f"AI summary unavailable: {last_error or 'all model attempts failed'}"

