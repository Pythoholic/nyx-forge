"""Local Ollama integration for structured creative prompt concepts."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
import requests
from pydantic import BaseModel, Field

from .engine_bridge import (
    build_semantic_context,
    generate_from_semantic,
    normalize_semantic_response,
)
from .prompt_engine import CompiledPrompt, ValidationResult
from .prompt_semantics import CanonicalPromptSpec, CreativeScene, SemanticPromptResponse
from .novelty import (
    NoveltyContext,
    maximum_similarity,
    provider_novelty_instruction,
    subject_variety_instruction,
)

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
SCENE_WORD_BUDGETS = {
    "subject": 16,
    "wardrobe": 10,
    "action": 12,
    "environment": 16,
    "lighting": 10,
    "mood": 8,
}
CONCISE_SCENE_WORD_BUDGETS = {
    "subject": 12,
    "wardrobe": 7,
    "action": 8,
    "environment": 10,
    "lighting": 6,
    "mood": 5,
}
NEGATIVE_DETAIL_WORD_LIMIT = 4


class TransformSuggestion(BaseModel):
    """One concise value suitable for a ForgeIMG contextual field."""

    suggestion: str = Field(min_length=3, max_length=180)


def limit_words(value: str, maximum: int) -> str:
    """Keep an LLM field concise without touching deterministic prompt tags."""
    words = value.strip(" ,.;\n\t").split()
    return " ".join(words[:maximum]).strip(" ,.;")


@dataclass(frozen=True, slots=True)
class OllamaPromptResult:
    """Final normalized prompt pair and the engine that produced it."""

    prompt: str
    negative_prompt: str
    model: str
    novelty_signature: str | None = None
    similarity_score: float = 0.0
    retry_count: int = 0
    semantic: CanonicalPromptSpec | None = None
    validation: ValidationResult | None = None
    repair_notes: tuple[str, ...] = ()
    compiled: CompiledPrompt | None = None


class OllamaPromptClient:
    """Small requests-based client for Ollama's local structured-output API."""

    def available_models(self) -> tuple[str, ...]:
        """Return installed text-generation models, excluding embeddings."""
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=(1.0, 3.0))
        response.raise_for_status()
        models = response.json().get("models", [])
        names: list[str] = []
        for item in models:
            details = item.get("details") or {}
            families = " ".join(details.get("families") or []).lower()
            family = str(details.get("family", "")).lower()
            name = str(item.get("name") or item.get("model") or "")
            if name and "bert" not in f"{family} {families}" and "embed" not in name.lower():
                names.append(name)
        return tuple(names)

    def suggest_transform(self, *, ollama_model: str, kind: str, image_base64: str) -> str:
        """Inspect the source and generate one compatible transform description."""
        schema = TransformSuggestion.model_json_schema()
        if kind == "background":
            instruction = (
                "Inspect the source image first. Propose one plausible replacement background behind the existing "
                "subject that matches the camera angle, perspective, lighting direction, and overall mood. Preserve "
                "the foreground subject exactly. Do not propose underwater, outer-space, fire, floating, or other "
                "subject-altering environments unless the source already depicts that context."
            )
        elif kind == "scene":
            instruction = (
                "Inspect the source identity first. Propose one complete new photographic scene for the same adult "
                "person, including an action, environment, lighting, and suitable clothing. Keep the person's "
                "gender presentation and defining visual character consistent. The entire head and face must remain "
                "clearly visible; avoid distant subjects and head crops."
            )
        else:
            instruction = (
                "Inspect the source image first, including pose, visible garment area, setting, lighting, and palette. "
                "Propose one fully clothed replacement outfit that fits the pose and occasion while preserving the "
                "person, face, hair, body proportions, and background exactly."
            )
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": ollama_model,
                "messages": [{
                    "role": "user",
                    "content": (
                        f"{instruction} Return only the replacement description, without analysis, a person, "
                        "camera instructions, quality tags, or introductory wording. Use 4 to 16 words. "
                        f"JSON schema: {json.dumps(schema, separators=(',', ':'))}"
                    ),
                    "images": [image_base64],
                }],
                "stream": False,
                "think": False,
                "format": schema,
                "keep_alive": 0,
                "options": {"temperature": 1.0, "top_p": 0.92, "num_predict": 80},
            },
            timeout=(3.0, 45.0),
        )
        response.raise_for_status()
        return TransformSuggestion.model_validate_json(response.json()["message"]["content"]).suggestion.strip()

    def generate(
        self,
        *,
        ollama_model: str,
        forge_model: str,
        style: str,
        rating: str,
        aspect_ratio: str,
        high_res: bool,
        orientation: str = "mixed",
        novelty: NoveltyContext | None = None,
        _retry: bool = False,
        _semantic_seed: int | None = None,
    ) -> OllamaPromptResult:
        """Select canonical concepts and compile them through Prompt Engine V1."""
        context = build_semantic_context(
            forge_model,
            rating,
            orientation,
            style=style,
            novelty_text=novelty.brief.as_json() if novelty is not None else "",
            seed=_semantic_seed,
        )
        schema = SemanticPromptResponse.model_json_schema()
        catalog = context.catalog_payload()
        system = (
            "You are a visual concept selector. Return only data matching the supplied JSON "
            "schema. Select concept IDs only from the supplied ForgeAI catalog. Do not write "
            "prompt prose, tags, negative prompts, or model syntax. ForgeAI validates and "
            "serializes the selected concepts."
        )
        prompt = (
            f"Create a coherent concept for checkpoint '{context.checkpoint}', Prompt Engine "
            f"profile '{context.model_id}', and "
            f"style '{style}'. Content rating: {context.rating.value}. Canvas: {aspect_ratio}. "
            f"Required orientation: {context.orientation.value}. "
            f"High-resolution second pass: {'enabled' if high_res else 'disabled'}. "
            f"{subject_variety_instruction()} "
            "Choose between 3 and 12 facet-to-ID mappings. Echo the required orientation exactly. "
            "Favor a compatible pose, environment, camera, and lighting combination. "
            f"Allowed catalog: {json.dumps(catalog, separators=(',', ':'))}. "
            f"JSON schema: {json.dumps(schema, separators=(',', ':'))}"
        )
        if novelty is not None:
            prompt = f"{provider_novelty_instruction(novelty, retry=_retry)} {prompt}"
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": ollama_model,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "format": schema,
                # Prompt generation and diffusion commonly share one GPU.
                # Release the LLM before Forge allocates its diffusion model.
                "keep_alive": 0,
                "options": {
                    "temperature": 1.0 if novelty and novelty.creativity_level == "experimental" else 0.9,
                    "top_p": 0.92,
                    "num_predict": 350,
                },
            },
            timeout=(3.0, 75.0),
        )
        response.raise_for_status()
        proposal = SemanticPromptResponse.model_validate_json(response.json()["response"])
        semantic = normalize_semantic_response(proposal, context)
        generated = generate_from_semantic(
            forge_model,
            semantic,
            seed=context.seed,
        )
        similarity = maximum_similarity(
            generated.compiled.positive,
            novelty.recent_prompts if novelty is not None else (),
        )
        result = OllamaPromptResult(
            generated.compiled.positive,
            generated.compiled.negative,
            ollama_model,
            novelty.brief.as_json() if novelty is not None else None,
            similarity,
            int(_retry),
            generated.semantic,
            generated.compiled.validation,
            generated.repair_notes,
            generated.compiled,
        )
        if (
            novelty is not None
            and novelty.recent_prompts
            and similarity > novelty.similarity_threshold
            and not _retry
        ):
            retry_result = self.generate(
                ollama_model=ollama_model,
                forge_model=forge_model,
                style=style,
                rating=rating,
                aspect_ratio=aspect_ratio,
                high_res=high_res,
                orientation=context.orientation.value,
                novelty=novelty,
                _retry=True,
                _semantic_seed=context.seed,
            )
            selected = min((result, retry_result), key=lambda item: item.similarity_score)
            return replace(selected, retry_count=1)
        return result
