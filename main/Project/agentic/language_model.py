import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from config import config

load_dotenv(config.ENV_FILE)
generation_settings = types.GenerateContentConfig(
    max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS, temperature=config.GEMINI_TEMPERATURE,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))


def call_language_model(prompt_text, agent_name):
    api_key, model_name = os.getenv(config.GEMINI_API_KEY_VARIABLE), os.getenv(config.GEMINI_MODEL_VARIABLE)
    if not api_key or not model_name:
        raise ValueError(f"{config.GEMINI_API_KEY_VARIABLE} and {config.GEMINI_MODEL_VARIABLE} must both be set "
                         f"in {config.ENV_FILE} before {agent_name} can call the language model")
    with genai.Client(api_key=api_key) as client:
        started = time.perf_counter()
        reply = client.models.generate_content(model=model_name, contents=prompt_text, config=generation_settings)
        latency = round((time.perf_counter() - started) * config.MILLISECONDS_PER_SECOND)
    usage = reply.usage_metadata
    finished = reply.candidates and reply.candidates[0].finish_reason == types.FinishReason.STOP
    if not (reply.text or "").strip() or not finished:
        raise ValueError(f"{agent_name} got an empty or unfinished reply from {model_name}")
    if usage is None or not usage.prompt_token_count or not usage.candidates_token_count:
        raise ValueError(f"{agent_name} got a reply from {model_name} that does not say how many tokens it used")
    return {"response_text": reply.text.strip(),
            "call_record": {"agent_name": agent_name, "model_name": model_name,
                            "prompt_tokens": usage.prompt_token_count, "response_tokens": usage.candidates_token_count,
                            "total_tokens": usage.total_token_count, "latency_milliseconds": latency}}


def _label_of(text):
    return " ".join(re.sub(config.RESPONSE_DECORATION_EDGE_PATTERN, "", text).replace("-", " ").upper().split())


def parse_labelled_response(response_text, field_labels, agent_name):
    parts, current = {label: [] for label in field_labels}, None
    for line in response_text.replace("**", "").splitlines():
        line = line.strip()
        head, colon, rest = line.partition(":")
        if colon and _label_of(head) in parts:
            current = _label_of(head)
            line = rest.strip(config.RESPONSE_DECORATION_CHARACTERS)
        if current is not None and line.strip(config.RESPONSE_DECORATION_CHARACTERS):
            parts[current].append(line)
    fields = {label: "\n".join(parts[label]).strip() for label in field_labels}
    missing = [label for label, text in fields.items() if not text]
    if missing:
        raise ValueError(f"{agent_name} replied without {', '.join(missing)}")
    return fields


def join_labelled_response(labelled_fields):
    return "\n\n".join(f"{label}: {text}" for label, text in labelled_fields.items())
