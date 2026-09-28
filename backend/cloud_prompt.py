"""Ephemeral cloud clients for structured prompt concepts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests

from .engine_bridge import (
    build_semantic_context,
    generate_from_semantic,
    normalize_semantic_response,
)
from .ollama_prompt import (
    TransformSuggestion,
)
from .prompt_engine import CompiledPrompt, PromptEngineError, ValidationResult
from .prompt_semantics import (
    CanonicalPromptSpec, SemanticPromptContext, SemanticPromptError,
    SemanticPromptResponse,
)
from .novelty import (
    NoveltyContext,
    maximum_similarity,
    provider_novelty_instruction,
    subject_variety_instruction,
)

ANTHROPIC_VERSION = "2023-06-01"
VENICE_MAX_COMPLETION_TOKENS = 180
PROVIDER_URLS = {
    "deepseek": "https://api.deepseek.com",
    "anthropic": "https://api.anthropic.com",
    "venice": "https://api.venice.ai/api/v1",
}
PREFERRED_MODELS = {
    "deepseek": ("deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat"),
    "anthropic": (
        "claude-haiku-4-5",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
    ),
    "venice": ("venice-uncensored-1-2",),
}


class CloudPromptProvider:
    """Provider-specific HTTP dialect behind the normalized semantic contract."""

    id: str
    base_url: str

    def headers(self, api_key: str) -> dict[str, str]:
        raise NotImplementedError

    def models_request(self) -> tuple[str, dict[str, int] | None]:
        raise NotImplementedError

    def prompt_request(
        self, model: str, system: str, prompt: str, *, temperature: float,
        max_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    def output_text(self, body: dict[str, Any]) -> str:
        raise NotImplementedError


class DeepSeekPromptProvider(CloudPromptProvider):
    id = "deepseek"
    base_url = PROVIDER_URLS[id]

    def headers(self, api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def models_request(self) -> tuple[str, dict[str, int] | None]:
        return f"{self.base_url}/models", None

    def prompt_request(
        self, model: str, system: str, prompt: str, *, temperature: float,
        max_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        return f"{self.base_url}/chat/completions", {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def output_text(self, body: dict[str, Any]) -> str:
        return str(body["choices"][0]["message"]["content"])


class AnthropicPromptProvider(CloudPromptProvider):
    id = "anthropic"
    base_url = PROVIDER_URLS[id]

    def headers(self, api_key: str) -> dict[str, str]:
        return {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def models_request(self) -> tuple[str, dict[str, int] | None]:
        return f"{self.base_url}/v1/models", {"limit": 100}

    def prompt_request(
        self, model: str, system: str, prompt: str, *, temperature: float,
        max_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        return f"{self.base_url}/v1/messages", {
            "model": model,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def output_text(self, body: dict[str, Any]) -> str:
        return str(next(
            block["text"]
            for block in body["content"]
            if isinstance(block, dict) and block.get("type") == "text"
        ))


class VenicePromptProvider(CloudPromptProvider):
    """Venice's OpenAI-compatible text API with its provider-specific limits."""

    id = "venice"
    base_url = PROVIDER_URLS[id]

    def headers(self, api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def models_request(self) -> tuple[str, dict[str, int] | None]:
        return f"{self.base_url}/models", None

    def prompt_request(
        self, model: str, system: str, prompt: str, *, temperature: float,
        max_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        del temperature
        # HARD COST LIMIT: Venice once emitted about 164K completion tokens.
        # Its API honors max_completion_tokens (not max_tokens), so every use
        # remains bounded even if a caller requests a larger shared limit.
        completion_limit = min(max_tokens, VENICE_MAX_COMPLETION_TOKENS)
        return f"{self.base_url}/chat/completions", {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_completion_tokens": completion_limit,
            "n": 1,
            "venice_parameters": {
                "include_venice_system_prompt": False,
                "enable_web_search": "off",
                "enable_web_scraping": False,
                "enable_web_citations": False,
            },
        }

    def output_text(self, body: dict[str, Any]) -> str:
        return str(body["choices"][0]["message"]["content"])


CLOUD_PROMPT_PROVIDERS: dict[str, CloudPromptProvider] = {
    provider.id: provider
    for provider in (
        DeepSeekPromptProvider(), AnthropicPromptProvider(), VenicePromptProvider()
    )
}


class CloudPromptError(RuntimeError):
    """A sanitized cloud-provider failure safe to expose to the UI."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class CloudCredentialResult:
    """Validated provider metadata; never contains the submitted secret."""

    provider: str
    models: tuple[str, ...]
    recommended: str


@dataclass(frozen=True, slots=True)
class CloudPromptResult:
    """Normalized prompt pair generated by a cloud language model."""

    prompt: str
    negative_prompt: str
    provider: str
    model: str
    novelty_signature: str | None = None
    similarity_score: float = 0.0
    retry_count: int = 0
    semantic: CanonicalPromptSpec | None = None
    validation: ValidationResult | None = None
    repair_notes: tuple[str, ...] = ()
    compiled: CompiledPrompt | None = None


def _provider_headers(provider: str, api_key: str) -> dict[str, str]:
    try:
        return CLOUD_PROMPT_PROVIDERS[provider].headers(api_key)
    except KeyError as exc:
        raise CloudPromptError("Unsupported cloud provider.", 422) from exc


def _checked_json(response: requests.Response, provider: str) -> dict[str, Any]:
    """Return provider JSON while keeping response bodies and secrets private."""
    if response.status_code in {401, 403}:
        raise CloudPromptError(f"{provider.title()} rejected the API key.", 401)
    if response.status_code == 429:
        raise CloudPromptError(f"{provider.title()} rate limit reached. Try again shortly.", 429)
    if response.status_code == 402:
        raise CloudPromptError(f"{provider.title()} reports a billing or credit issue.", 402)
    if not response.ok:
        raise CloudPromptError(
            f"{provider.title()} request failed (HTTP {response.status_code})."
        )
    try:
        body = response.json()
    except requests.JSONDecodeError as exc:
        raise CloudPromptError(f"{provider.title()} returned invalid JSON.") from exc
    if not isinstance(body, dict):
        raise CloudPromptError(f"{provider.title()} returned an invalid response.")
    return body


def _json_object(value: str) -> str:
    """Extract one JSON object from plain or fenced model output."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise CloudPromptError("The cloud model did not return a structured prompt.")
    return cleaned[start : end + 1]


def _request_structured_scene(
    *,
    provider: CloudPromptProvider,
    url: str,
    headers: dict[str, str],
    request_body: dict[str, Any],
) -> SemanticPromptResponse:
    """Request and validate one scene while keeping provider output private."""
    try:
        response = requests.post(
            url,
            headers=headers,
            json=request_body,
            timeout=(5.0, 75.0),
        )
    except requests.Timeout as exc:
        raise CloudPromptError(f"{provider.id.title()} prompt request timed out.", 504) from exc
    except requests.RequestException as exc:
        raise CloudPromptError(f"Could not reach {provider.id.title()}.") from exc
    body = _checked_json(response, provider.id)
    try:
        output = provider.output_text(body)
        if not output or not str(output).strip():
            raise ValueError("empty structured content")
        return SemanticPromptResponse.model_validate_json(_json_object(str(output)))
    except (KeyError, IndexError, StopIteration, TypeError, ValueError) as exc:
        raise CloudPromptError(
            f"{provider.id.title()} did not return a valid structured prompt."
        ) from exc


def _scene_instructions(
    *,
    context: SemanticPromptContext,
    aspect_ratio: str,
    high_res: bool,
    novelty: NoveltyContext | None = None,
    retry: bool = False,
) -> tuple[str, str]:
    """Build the provider-neutral canonical concept-selection request."""
    schema = SemanticPromptResponse.model_json_schema()
    system = (
        "You are a visual concept selector. Return only one JSON object matching the supplied "
        "schema. Select concept IDs only from the supplied ForgeAI catalog. Do not write prompt "
        "prose, tags, negative prompts, or checkpoint syntax. ForgeAI owns validation and serialization."
    )
    prompt = (
        f"Checkpoint: {context.checkpoint}. Prompt Engine profile: {context.model_id}. "
        f"Style: {context.style_parent}. "
        f"Content rating: {context.rating.value}. Required orientation: {context.orientation.value}. "
        f"Canvas: {aspect_ratio}. High-resolution pass: {'yes' if high_res else 'no'}. "
        f"{subject_variety_instruction()} "
        "Choose between 3 and 12 facet-to-ID mappings and echo the required orientation exactly. "
        "Favor a compatible pose, environment, camera, and lighting combination. Allowed catalog: "
        f"{json.dumps(context.catalog_payload(), separators=(',', ':'))}. JSON schema: "
        f"{json.dumps(schema, separators=(',', ':'))}."
    )
    if novelty is not None:
        prompt = f"{provider_novelty_instruction(novelty)} {prompt}"
    if retry:
        prompt = (
            "This is a recovery attempt. Return complete valid JSON and substantially vary any "
            f"repetitive staging. {prompt}"
        )
    return system, prompt


class CloudPromptClient:
    """Stateless provider client; API keys are used only for the active request."""

    def validate_credentials(self, provider: str, api_key: str) -> CloudCredentialResult:
        """Authenticate by listing accessible models without spending generation tokens."""
        implementation = CLOUD_PROMPT_PROVIDERS.get(provider)
        if implementation is None:
            raise CloudPromptError("Unsupported cloud provider.", 422)
        url, params = implementation.models_request()
        try:
            response = requests.get(
                url,
                headers=implementation.headers(api_key),
                params=params,
                timeout=(3.0, 15.0),
            )
        except requests.Timeout as exc:
            raise CloudPromptError(f"{provider.title()} credential check timed out.", 504) from exc
        except requests.RequestException as exc:
            raise CloudPromptError(f"Could not reach {provider.title()}.") from exc
        body = _checked_json(response, provider)
        models = tuple(
            str(item["id"])
            for item in body.get("data", [])
            if isinstance(item, dict) and item.get("id")
        )
        if not models:
            raise CloudPromptError(f"{provider.title()} returned no available models.")
        preferred = PREFERRED_MODELS[provider]
        recommended = next(
            (
                model
                for preferred_name in preferred
                for model in models
                if model == preferred_name or model.startswith(f"{preferred_name}-")
            ),
            models[0],
        )
        return CloudCredentialResult(provider, models, recommended)

    def suggest_transform(
        self, *, provider: str, api_key: str, cloud_model: str, kind: str
    ) -> str:
        """Generate one bounded background or outfit description."""
        implementation = CLOUD_PROMPT_PROVIDERS.get(provider)
        if implementation is None:
            raise CloudPromptError("Unsupported cloud provider.", 422)
        subject = (
            "replacement background"
            if kind == "background"
            else "complete photographic scene for the same adult person, including action, environment, lighting, and clothing"
            if kind == "scene"
            else "fully clothed outfit"
        )
        schema = TransformSuggestion.model_json_schema()
        system = "Return only valid JSON. Be visual, concise, and safe."
        prompt = (
            f"Invent one distinctive {subject} for image editing. Return only the description, "
            "without a person, camera instructions, quality tags, or introductory wording. "
            f"Use 4 to 14 words. JSON schema: {json.dumps(schema, separators=(',', ':'))}"
        )
        headers = implementation.headers(api_key)
        url, request_body = implementation.prompt_request(
            cloud_model, system, prompt, temperature=1.0, max_tokens=120
        )
        try:
            response = requests.post(url, headers=headers, json=request_body, timeout=(5.0, 45.0))
        except requests.Timeout as exc:
            raise CloudPromptError(f"{provider.title()} suggestion request timed out.", 504) from exc
        except requests.RequestException as exc:
            raise CloudPromptError(f"Could not reach {provider.title()}.") from exc
        body = _checked_json(response, provider)
        try:
            output = implementation.output_text(body)
            return TransformSuggestion.model_validate_json(_json_object(str(output))).suggestion.strip()
        except (KeyError, IndexError, StopIteration, TypeError, ValueError) as exc:
            raise CloudPromptError(f"{provider.title()} did not return a valid suggestion.") from exc

    def generate(
        self,
        *,
        provider: str,
        api_key: str,
        cloud_model: str,
        forge_model: str,
        style: str,
        rating: str,
        aspect_ratio: str,
        high_res: bool,
        orientation: str = "mixed",
        novelty: NoveltyContext | None = None,
    ) -> CloudPromptResult:
        """Select canonical concepts and compile them through Prompt Engine V1."""
        implementation = CLOUD_PROMPT_PROVIDERS.get(provider)
        if implementation is None:
            raise CloudPromptError("Unsupported cloud provider.", 422)
        if not cloud_model:
            raise CloudPromptError("Choose a cloud prompt model.", 422)
        try:
            context = build_semantic_context(
                forge_model,
                rating,
                orientation,
                style=style,
                novelty_text=novelty.brief.as_json() if novelty is not None else "",
            )
        except (PromptEngineError, ValueError) as exc:
            raise CloudPromptError("The selected settings are not supported by Prompt Engine.", 422) from exc
        headers = implementation.headers(api_key)
        candidates: list[
            tuple[float, CanonicalPromptSpec, CompiledPrompt, tuple[str, ...]]
        ] = []
        last_error: CloudPromptError | None = None
        # DeepSeek documents that JSON mode can occasionally return empty
        # content. One bounded recovery request also serves the novelty retry.
        attempts = 2
        for attempt in range(attempts):
            system, prompt = _scene_instructions(
                context=context,
                aspect_ratio=aspect_ratio,
                high_res=high_res,
                novelty=novelty,
                retry=attempt > 0,
            )
            url, request_body = implementation.prompt_request(
                cloud_model,
                system,
                prompt,
                temperature=(
                    1.0 if novelty and novelty.creativity_level == "experimental" else 0.9
                ),
                max_tokens=800,
            )
            try:
                proposal = _request_structured_scene(
                    provider=implementation,
                    url=url,
                    headers=headers,
                    request_body=request_body,
                )
            except CloudPromptError as exc:
                last_error = exc
                if candidates:
                    break
                if attempt + 1 < attempts and exc.status_code not in {401, 402, 403, 429}:
                    continue
                raise
            try:
                semantic = normalize_semantic_response(proposal, context)
                generated = generate_from_semantic(
                    forge_model,
                    semantic,
                    seed=context.seed,
                )
            except (SemanticPromptError, PromptEngineError) as exc:
                # Concept conflicts are repaired locally by Prompt Engine. If
                # its bounded repair cannot produce a valid AST, do not ask
                # the LLM to rewrite repeatedly until validation happens to pass.
                raise CloudPromptError(
                    f"{provider.title()} returned unusable canonical concepts."
                ) from exc
            similarity = maximum_similarity(
                generated.compiled.positive,
                novelty.recent_prompts if novelty is not None else (),
            )
            candidates.append((
                similarity,
                generated.semantic,
                generated.compiled,
                generated.repair_notes,
            ))
            if novelty is None or similarity <= novelty.similarity_threshold:
                break

        if not candidates:
            raise last_error or CloudPromptError(
                f"{provider.title()} did not return a valid structured prompt."
            )
        similarity, semantic, compiled_value, repair_notes = min(
            candidates, key=lambda item: item[0]
        )
        compiled = compiled_value
        return CloudPromptResult(
            compiled.positive,
            compiled.negative,
            provider,
            cloud_model,
            (
                novelty.brief.as_json(canonical_concept_ids=semantic.concept_ids)
                if novelty is not None else None
            ),
            similarity,
            len(candidates) - 1,
            semantic,
            compiled.validation,
            repair_notes,
            compiled,
        )
