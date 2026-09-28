"""Wan-specific prompt construction; intentionally separate from SDXL tags."""
from __future__ import annotations

import random
import json
import re

import requests

SDXL_ONLY = re.compile(
    r"(?:^|,\s*)(?:score_\d+(?:_up)?|source_[\w-]+|rating_[\w-]+|1woman|1girl|1boy|solo|female focus|male focus|[()]?medium[()]?|BREAK)(?=,|$)",
    re.IGNORECASE,
)

MOTION_LANGUAGE = re.compile(
    r"\b(?:animate[sd]?|blinks?|breathes?|camera|changes?|doll(?:y|ies)|"
    r"drift(?:s|ing)?|flickers?|flows?|moves?|motion|orbits?|paces?|pans?|"
    r"push(?:es)?|reaches?|runs?|shifts?|sways?|tracks?|turns?|walks?|waves?|zooms?)\b",
    re.IGNORECASE,
)

VIDEO_PROMPT_PROFILES = {
    "actions": (
        "walks forward slowly for the duration of the shot",
        "turns gradually toward the camera and pauses",
        "looks around naturally, then takes two measured steps",
        "reaches toward a nearby object with deliberate movement",
    ),
    "camera": (
        "the camera makes a slow steady push forward",
        "the camera remains locked with subtle cinematic parallax",
        "the camera pans gently from left to right",
        "the camera tracks the subject smoothly at eye level",
    ),
    "secondary": (
        "hair and fabric move softly in a light breeze",
        "background light shifts gradually across the scene",
        "small environmental details move naturally over time",
        "shadows and reflections change subtly with the motion",
    ),
    "pace": ("slow and continuous", "natural and unhurried", "gradual with no sudden cuts"),
}


def strip_sdxl_tags(prompt: str) -> str:
    """Turn an image prompt into useful prose without leaking checkpoint tags."""
    cleaned = SDXL_ONLY.sub("", prompt.replace(r"\(medium\)", "medium"))
    cleaned = re.sub(r"\s*,\s*,+", ", ", cleaned)
    return cleaned.strip(" ,")


def has_motion_language(prompt: str) -> bool:
    """Return whether a prompt describes change over time rather than appearance only."""
    return bool(MOTION_LANGUAGE.search(prompt))


def validate_video_prompt(prompt: str) -> str:
    """Reject checkpoint-tag or appearance-only text before it reaches Wan."""
    cleaned = strip_sdxl_tags(prompt)
    if cleaned != prompt.strip(" ,"):
        raise ValueError("Motion prompts cannot contain Stable Diffusion or Danbooru tags. Generate a motion prompt first.")
    if not has_motion_language(cleaned):
        raise ValueError("This describes appearance but not movement. Generate or enter a motion prompt first.")
    return cleaned


def curated_video_prompt(*, mode: str, source_prompt: str = "") -> str:
    rng = random.SystemRandom()
    base = strip_sdxl_tags(source_prompt)
    subject = base or "A cinematic subject in a coherent detailed environment"
    return ". ".join((subject, rng.choice(VIDEO_PROMPT_PROFILES["actions"]), rng.choice(VIDEO_PROMPT_PROFILES["camera"]), rng.choice(VIDEO_PROMPT_PROFILES["secondary"]), f"Movement is {rng.choice(VIDEO_PROMPT_PROFILES['pace'])}")) + "."


VIDEO_SYSTEM_INSTRUCTION = (
    "Write one concise Wan 2.1 video prompt in natural language. Describe what the subject does over time, "
    "camera movement, secondary motion such as hair, cloth, water or light, and pacing. Return only the prompt. "
    "Never use Stable Diffusion score, source, rating, BREAK, weighting, or Danbooru tags."
)

VIDEO_VISION_INSTRUCTION = (
    f"{VIDEO_SYSTEM_INSTRUCTION} Analyze the supplied starting frame first. Preserve the visible subject, "
    "composition, and setting. Propose motion that is physically plausible for what is actually visible; "
    "do not invent a different subject or restate appearance tags."
)


def ollama_video_prompt(model: str, *, context: str = "", image_base64: str | None = None) -> str:
    payload = {
        "model": model,
        "system": VIDEO_VISION_INSTRUCTION if image_base64 else VIDEO_SYSTEM_INSTRUCTION,
        "prompt": (
            "Study this starting frame and write motion that naturally continues it."
            if image_base64
            else f"Create motion for this starting concept: {strip_sdxl_tags(context) or 'invent a cinematic scene'}"
        ),
        "stream": False,
        "think": False,
        "keep_alive": 0,
        "options": {"temperature": 0.75 if image_base64 else 0.9, "top_p": 0.9, "num_predict": 180},
    }
    if image_base64:
        payload["images"] = [image_base64]
    response = requests.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=(3, 180))
    response.raise_for_status()
    return str(response.json()["response"]).strip()


def cloud_video_prompt(provider: str, api_key: str, model: str, *, context: str = "") -> str:
    prompt = f"{VIDEO_SYSTEM_INSTRUCTION}\nStarting concept: {strip_sdxl_tags(context) or 'invent a cinematic scene'}"
    if provider == "deepseek":
        response = requests.post("https://api.deepseek.com/chat/completions", headers={"Authorization": f"Bearer {api_key}"}, json={
            "model": model, "messages": [{"role": "system", "content": VIDEO_SYSTEM_INSTRUCTION}, {"role": "user", "content": prompt}],
            "stream": False, "temperature": 0.9, "max_tokens": 220,
        }, timeout=(5, 75))
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"]).strip()
    if provider == "anthropic":
        response = requests.post("https://api.anthropic.com/v1/messages", headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"}, json={
            "model": model, "system": VIDEO_SYSTEM_INSTRUCTION, "messages": [{"role": "user", "content": prompt}], "max_tokens": 220,
        }, timeout=(5, 75))
        response.raise_for_status()
        return str(next(block["text"] for block in response.json()["content"] if block.get("type") == "text")).strip()
    raise ValueError("Unsupported cloud provider.")

# --- Image-aware analysis -------------------------------------------------
#
# Prompts written without seeing the frame produce impossible motion: "walks
# forward" for a subject lying down. And the camera embedding was chosen
# independently of the prompt, so the two conditioning signals disagreed.
#
# The vision model returns strict JSON describing what is actually in the
# frame; code then validates it and composes the prompt, so an invalid camera
# or an implausible action can be rejected before spending minutes generating.

CAMERA_POSES = (
    "Static", "Pan Up", "Pan Down", "Pan Left", "Pan Right",
    "Zoom In", "Zoom Out", "Anti Clockwise (ACW)", "ClockWise (CW)",
)

# Camera geometry the frame cannot support. A close or flat frame has no
# parallax for an orbit; a subject already lying down cannot be tilted up to.
CAMERA_BLOCKLIST = {
    "lying": {"Pan Up", "ClockWise (CW)", "Anti Clockwise (ACW)"},
    "seated": {"Pan Up"},
}

# The canonical sentence for each pose. Appended only when the caller wants the
# text to restate the camera move; the embedding owns the geometry either way.
CAMERA_SENTENCE = {
    "Static": "The camera holds still.",
    "Pan Up": "The camera tilts upward slowly.",
    "Pan Down": "The camera tilts downward slowly.",
    "Pan Left": "The camera pans slowly left.",
    "Pan Right": "The camera pans slowly right.",
    "Zoom In": "The camera slowly pushes in.",
    "Zoom Out": "The camera slowly pulls back.",
    "Anti Clockwise (ACW)": "The camera orbits slowly counter-clockwise around the subject.",
    "ClockWise (CW)": "The camera orbits slowly clockwise around the subject.",
}

VISION_ANALYSIS_INSTRUCTION = (
    "Analyse this starting frame for a short video. "
    "Reply with ONLY a JSON object and no markdown fence: "
    '{"pose":"standing|seated|lying|moving","facing":"front|left|right|away",'
    '"motion_candidates":[{"action":"short phrase describing movement",'
    '"intensity":"subtle|moderate|large"}],'
    '"secondary_motion":["hair, cloth, water or light movement already visible"],'
    '"camera":{"recommended":"Static|Pan Up|Pan Down|Pan Left|Pan Right|Zoom In|'
    'Zoom Out|Anti Clockwise (ACW)|ClockWise (CW)","reason":"short"}}. '
    "Only propose motion that is physically possible for the visible pose. "
    "Do not invent a different subject."
)


def analyse_source_frame(model: str, image_base64: str, *, timeout: float = 180.0) -> dict:
    """Return validated structured analysis of a starting frame.

    Raises ValueError if the model does not return usable JSON, so the caller
    can fall back rather than compose a prompt from nothing.
    """
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": model,
            "prompt": VISION_ANALYSIS_INSTRUCTION,
            "images": [image_base64],
            "stream": False,
            "think": False,
            # The GPU is shared with Wan sampling, which peaks near the 10 GB
            # ceiling, so the vision model must not stay resident.
            "keep_alive": 0,
            "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 400},
        },
        timeout=(3, timeout),
    )
    response.raise_for_status()
    return _parse_analysis(str(response.json().get("response") or ""))


def _parse_analysis(raw: str) -> dict:
    """Validate and normalise the vision model's JSON."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Vision model did not return JSON.")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Vision model returned invalid JSON: {exc}") from exc

    pose = str(data.get("pose") or "").lower()
    if pose not in {"standing", "seated", "lying", "moving"}:
        pose = "standing"
    actions = [
        str(item.get("action") or "").strip()
        for item in data.get("motion_candidates") or []
        if isinstance(item, dict) and str(item.get("action") or "").strip()
    ]
    if not actions:
        raise ValueError("Vision model proposed no motion.")
    secondary = [str(x).strip() for x in data.get("secondary_motion") or [] if str(x).strip()]
    camera = data.get("camera") if isinstance(data.get("camera"), dict) else {}
    recommended = str(camera.get("recommended") or "").strip()
    if recommended not in CAMERA_POSES or recommended in CAMERA_BLOCKLIST.get(pose, set()):
        recommended = "Zoom In" if pose in {"seated", "lying"} else "Static"
    return {
        "pose": pose,
        "facing": str(data.get("facing") or "front").lower(),
        "actions": actions[:2],
        "secondary_motion": secondary[:2],
        "camera": recommended,
        "camera_reason": str(camera.get("reason") or "").strip(),
    }


def compose_i2v_prompt(analysis: dict, *, include_camera_sentence: bool = True) -> str:
    """Build a Wan I2V prompt from validated analysis.

    The image already carries identity, clothing and composition, so this
    describes only what changes.
    """
    parts = [analysis["actions"][0].rstrip(".") + "."]
    if len(analysis["actions"]) > 1:
        parts.append(analysis["actions"][1].rstrip(".") + ".")
    if analysis["secondary_motion"]:
        parts.append(", ".join(analysis["secondary_motion"]).rstrip(".") + ".")
    if include_camera_sentence:
        parts.append(CAMERA_SENTENCE[analysis["camera"]])
    parts.append("Motion is continuous with no cut; identity and composition stay stable.")
    return " ".join(part[0].upper() + part[1:] for part in parts if part)
