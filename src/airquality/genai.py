from __future__ import annotations
import json
import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)
PROMPT_PATH = PROJECT_ROOT / "prompts" / "shift_handover_v1.txt"


def generate_handover_summary(prediction: float, unit: str, validation_mae: float,
                               recent_context: dict, model_name: str | None = None) -> tuple[str, dict]:
    """Optional LLM-generated handover note. This never changes the numerical forecast."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured. Set it only if you choose to use the optional LLM feature.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to enable the optional generative-model feature.") from exc
    model_name = model_name or os.getenv("GENAI_MODEL")
    if not model_name:
        raise RuntimeError("GENAI_MODEL is not configured. Set it to a currently supported model identifier before enabling this optional feature.")
    instructions = PROMPT_PATH.read_text(encoding="utf-8")
    prompt_sha256 = hashlib.sha256(instructions.encode("utf-8")).hexdigest()
    payload = {
        "forecast_benzene": round(float(prediction), 3),
        "unit": unit,
        "validation_mae": round(float(validation_mae), 3),
        "recent_sensor_context": recent_context,
        "important_limitations": [
            "This is a one-hour-ahead research model trained on historical data from one Italian roadside station in 2004-2005.",
            "The numerical forecast is produced by the LSTM ANN, not by this language model.",
            "Do not interpret the forecast as a regulatory threshold or medical/safety guarantee.",
            "Recommend a reference-analyser check when observations are unusual or data quality is poor."
        ]
    }
    client = OpenAI(api_key=api_key)
    import time
    start = time.perf_counter()
    response = client.responses.create(
        model=model_name,
        instructions=instructions,
        input=json.dumps(payload, ensure_ascii=False),
    )
    elapsed = time.perf_counter() - start
    text = getattr(response, "output_text", None) or "No text returned by the generative model."
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", None))
    output_tokens = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", None))
    total_tokens = getattr(usage, "total_tokens", None)
    try:
        input_rate = float(os.getenv("GENAI_INPUT_USD_PER_1M", ""))
        output_rate = float(os.getenv("GENAI_OUTPUT_USD_PER_1M", ""))
        cost_estimate = ((input_tokens or 0) * input_rate + (output_tokens or 0) * output_rate) / 1_000_000
    except (TypeError, ValueError):
        cost_estimate = None
    metadata = {
        "provider": "OpenAI API",
        "model": getattr(response, "model", None) or model_name,
        "requested_model": model_name,
        "response_id": getattr(response, "id", None),
        "prompt_sha256": prompt_sha256,
        "prompt_file": "prompts/shift_handover_v1.txt",
        "prompt_version": "shift_handover_v1",
        "latency_seconds": round(elapsed, 3),
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(cost_estimate, 8) if cost_estimate is not None else None,
        "pricing_rates_configured": cost_estimate is not None,
        "numeric_prediction_changed": False,
        "future_actual_target_supplied": False,
    }
    return text, metadata
