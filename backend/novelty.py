"""History-aware creative briefs and lightweight prompt similarity scoring."""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Protocol


CREATIVITY_LEVELS = {"consistent", "balanced", "experimental"}

# LLM-generated subjects default to youthful, slender descriptions unless
# explicitly steered otherwise. Picking one age/build pair per request keeps
# variety on par with the curated engine's own age and body-type pools.
AGE_RANGES = (
    "in her mid-20s",
    "in her late 20s",
    "in her early 30s",
    "in her mid-30s",
    "in her late 30s",
    "in her 40s",
    "in her late 40s",
    "in her 50s, a mature woman",
)
BODY_TYPES = (
    "an athletic build with balanced proportions",
    "a softly rounded build with a natural stance",
    "an hourglass build with balanced proportions",
    "a petite build with gentle curves",
    "a tall, statuesque build with long lines",
    "a full-figured build with natural proportions",
    "a toned build with strong posture",
    "a broad-shouldered build with a grounded stance",
    "a lean build with relaxed posture",
    "a pear-shaped build with balanced proportions",
)
# Without steering, LLMs converge on "long flowing hair" and "expressive
# eyes" for nearly every generated subject regardless of style or engine.
HAIR_DESCRIPTIONS = (
    "a short bob cut",
    "a sleek high ponytail",
    "shoulder-length waves",
    "a tight braid",
    "a messy bun",
    "chin-length hair with blunt bangs",
    "long straight hair",
    "voluminous curls",
    "an asymmetric pixie cut",
    "twin braids",
)
EYE_DESCRIPTIONS = (
    "sharp, narrow eyes",
    "large, round eyes",
    "heavy-lidded eyes",
    "almond-shaped eyes",
    "a calm, attentive gaze",
    "wide, alert eyes",
)


def subject_variety_instruction(rng: random.Random | None = None) -> str:
    """Return one randomly paired age, body-type, hair, and eye instruction."""
    source = rng or random.SystemRandom()
    age = source.choice(AGE_RANGES)
    body_type = source.choice(BODY_TYPES)
    hair = source.choice(HAIR_DESCRIPTIONS)
    eyes = source.choice(EYE_DESCRIPTIONS)
    return (
        f"The woman should be {age}, with this body type: {body_type}, "
        f"{hair}, and {eyes}. Do not default to a young adult with a slender "
        "build, long flowing hair, and generically 'expressive' eyes every "
        "time - vary age, body type, hairstyle, and eye shape across requests. "
        "Keep the description respectful, non-sexual, and focused on the "
        "character's overall silhouette and presence."
    )


class PromptHistoryItem(Protocol):
    """Minimal persisted prompt shape consumed by the novelty engine."""

    positive_prompt: str
    novelty_signature: str | None
    user_rating: int | None
    feedback_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SceneDirection:
    """A coherent location/action family used as the seed for a scene brief."""

    theme: str
    environments: tuple[str, ...]
    actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NoveltyBrief:
    """Structured creative direction shared by every prompt engine."""

    theme: str
    environment: str
    action: str
    lighting: str
    palette: str
    mood: str
    camera: str

    def as_json(self, *, canonical_concept_ids: Iterable[str] = ()) -> str:
        """Serialize a stable signature suitable for SQLite persistence."""
        payload: dict[str, object] = asdict(self)
        concept_ids = tuple(dict.fromkeys(canonical_concept_ids))
        if concept_ids:
            payload["canonical_concept_ids"] = concept_ids
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    def instruction(self) -> str:
        """Render the brief as concise provider-neutral art direction."""
        return (
            f"Theme: {self.theme}; location: {self.environment}; action: {self.action}; "
            f"lighting: {self.lighting}; palette: {self.palette}; mood: {self.mood}; "
            f"camera treatment: {self.camera}."
        )


@dataclass(frozen=True, slots=True)
class NoveltyContext:
    """One brief plus local-only history signals used to reject repetition."""

    brief: NoveltyBrief
    creativity_level: str
    avoid_summary: str
    quality_guidance: tuple[str, ...]
    recent_prompts: tuple[str, ...]
    similarity_threshold: float


SCENE_DIRECTIONS = (
    SceneDirection(
        "urban discovery",
        (
            "a rain-washed tram platform before sunrise",
            "a rooftop greenhouse above a dense modern city",
            "an empty brutalist plaza after a summer storm",
        ),
        ("crossing the frame against the wind", "studying a folded city map", "waiting as a tram approaches"),
    ),
    SceneDirection(
        "wilderness expedition",
        (
            "a windswept alpine observatory above the clouds",
            "a moss-covered footbridge in an ancient cedar forest",
            "a volcanic black-sand coast beneath distant cliffs",
        ),
        ("adjusting field equipment", "walking along a narrow ridge", "pausing beside a weathered trail marker"),
    ),
    SceneDirection(
        "coastal architecture",
        (
            "a modern cliffside house overlooking a rough sea",
            "a whitewashed island courtyard above a quiet harbor",
            "an abandoned lighthouse gallery during an incoming storm",
        ),
        ("opening tall shutters toward the sea", "descending broad stone steps", "watching distant storm clouds gather"),
    ),
    SceneDirection(
        "scientific exploration",
        (
            "a glass botanical laboratory filled with rare luminous plants",
            "a remote desert radio observatory under a vast sky",
            "an underwater research habitat with panoramic windows",
        ),
        ("cataloguing an unusual specimen", "calibrating a precision instrument", "reviewing handwritten field notes"),
    ),
    SceneDirection(
        "artisan workshop",
        (
            "a sunlit ceramic studio lined with unfinished vessels",
            "a traditional printmaking workshop with hanging paper",
            "a watchmaker's bench crowded with brass mechanisms",
        ),
        ("shaping a workpiece by hand", "examining a newly finished detail", "arranging tools before beginning work"),
    ),
    SceneDirection(
        "live performance",
        (
            "an empty opera stage moments before rehearsal",
            "a small jazz club after the audience has left",
            "a contemporary dance studio with mirrored walls",
        ),
        ("rehearsing a decisive movement", "stepping through a pool of stage light", "resting between performances"),
    ),
    SceneDirection(
        "retro futurism",
        (
            "a streamlined 1960s lunar lounge with curved windows",
            "a chrome overnight train crossing a glowing salt flat",
            "an analog mission-control room filled with colored displays",
        ),
        ("checking an illuminated control panel", "walking through a curved observation car", "listening to a vintage radio transmission"),
    ),
    SceneDirection(
        "historical intrigue",
        (
            "a candlelit Renaissance map archive",
            "a grand Art Deco railway concourse at midnight",
            "a weathered hilltop monastery library",
        ),
        ("unrolling an ancient chart", "hurrying beneath a monumental clock", "searching a wall of handwritten records"),
    ),
    SceneDirection(
        "seasonal ritual",
        (
            "a lantern-lit orchard during the first autumn frost",
            "a quiet flower market opening on a spring morning",
            "a snow-covered glass pavilion at blue hour",
        ),
        ("hanging the final paper lantern", "carrying an armful of fresh flowers", "brushing snow from a glass railing"),
    ),
    SceneDirection(
        "athletic movement",
        (
            "a sunken concrete swimming arena before opening",
            "a high-altitude running track surrounded by mountains",
            "an industrial fencing hall with tall north-facing windows",
        ),
        ("stretching before a solo training session", "accelerating through a sweeping turn", "holding a poised defensive stance"),
    ),
    SceneDirection(
        "surreal dreamscape",
        (
            "a flooded museum where paintings float above the water",
            "an endless mirrored garden under two pale moons",
            "a library whose staircases disappear into clouds",
        ),
        ("wading toward a suspended doorway", "following a trail of floating petals", "reaching for a book drifting overhead"),
    ),
    SceneDirection(
        "travel editorial",
        (
            "a sleeper-train compartment crossing a snowy valley",
            "a quiet airport terminal during a red-eye layover",
            "a vintage convertible stopped on a desert highway",
        ),
        ("watching the landscape pass the window", "walking past repeating pools of terminal light", "checking the route beside the open car door"),
    ),
)

PRIVATE_SCENE_DIRECTIONS = (
    SceneDirection(
        "private coastal retreat",
        (
            "a secluded cliffside villa terrace screened by stone walls",
            "a private island courtyard overlooking the sea",
            "an isolated glass pavilion above a rocky cove",
        ),
        ("standing against the sea breeze", "walking beside a long reflecting pool", "resting against a sun-warmed stone arch"),
    ),
    SceneDirection(
        "private creative studio",
        (
            "a closed artist loft filled with large unfinished canvases",
            "a private sculptor's studio with plaster forms and skylights",
            "an empty photography set built from translucent fabric panels",
        ),
        ("posing among unfinished works", "crossing through layered fabric panels", "turning beneath a broad overhead skylight"),
    ),
    SceneDirection(
        "secluded nature retreat",
        (
            "a hidden geothermal pool enclosed by volcanic rock",
            "a private forest bathing pavilion surrounded by mist",
            "an isolated desert courtyard beneath a star-filled sky",
        ),
        ("stepping from the water onto warm stone", "standing beneath a gentle outdoor shower", "stretching beside a low fire basin"),
    ),
    SceneDirection(
        "after-hours performance",
        (
            "a locked opera stage after the final rehearsal",
            "a private mirrored dance studio late at night",
            "an empty cabaret set behind closed theater doors",
        ),
        ("holding a poised theatrical stance", "moving through a slow dance phrase", "stepping into a single pool of stage light"),
    ),
    SceneDirection(
        "private glasshouse",
        (
            "a locked tropical conservatory after sunset",
            "a secluded winter garden beneath rain-streaked glass",
            "a private rooftop greenhouse hidden above the city",
        ),
        ("walking between broad tropical leaves", "touching condensation on a glass wall", "pausing beneath hanging vines"),
    ),
    SceneDirection(
        "fantasy sanctuary",
        (
            "a hidden moon temple accessible only by a stone bridge",
            "an enchanted mineral cavern surrounding a still pool",
            "a sealed celestial observatory above the clouds",
        ),
        ("approaching a luminous altar", "wading through shallow reflective water", "raising one hand toward a rotating star map"),
    ),
    SceneDirection(
        "architectural hideaway",
        (
            "a private desert house formed from curved rammed-earth walls",
            "a secluded alpine cabin with one panoramic glass facade",
            "a closed modernist bathhouse built around a central courtyard",
        ),
        (
            "crossing a band of geometric sunlight",
            "resting beside a sunken conversation pit",
            "opening a concealed wall toward the landscape",
        ),
    ),
    SceneDirection(
        "midnight transit",
        (
            "a private sleeper carriage moving through a moonlit valley",
            "an empty first-class airport lounge after the final departure",
            "a locked vintage ferry salon crossing dark water",
        ),
        (
            "watching lights pass across the window",
            "walking between rows of empty seats",
            "standing beside a polished brass doorway",
        ),
    ),
    SceneDirection(
        "ceremonial interior",
        (
            "a sealed marble rotunda illuminated by floor lanterns",
            "a private tea pavilion surrounded by a still black pond",
            "an abandoned palace music room prepared for one guest",
        ),
        (
            "arranging a final ceremonial object",
            "crossing the room with measured confidence",
            "standing beneath a suspended ring of candlelight",
        ),
    ),
    SceneDirection(
        "surreal private world",
        (
            "a mirrored chamber filled with slow-floating white feathers",
            "a flooded observatory beneath an artificial eclipse",
            "a silent indoor garden where glass flowers emit dim light",
        ),
        (
            "moving through overlapping reflections",
            "reaching toward a suspended sphere of light",
            "wading between luminous glass plants",
        ),
    ),
)

LIGHTING = (
    "cool predawn ambient light with a narrow warm practical",
    "hard midday sunlight broken by architectural shadows",
    "soft overcast daylight with delicate reflected fill",
    "low-key tungsten pools against deep neutral shadows",
    "silvery moonlight with restrained atmospheric haze",
    "late-afternoon side light with long directional shadows",
    "colored signage reflected through rain without neon overload",
    "high clerestory light with visible dust in the air",
)

PALETTES = (
    "slate blue, weathered silver, and muted amber",
    "bone white, charcoal, and one restrained red accent",
    "deep forest green, oxidized copper, and fog gray",
    "sandstone, faded indigo, and warm black",
    "cool cyan, brushed chrome, and desaturated orange",
    "burgundy, parchment, and dark walnut",
    "sea glass, chalk white, and storm blue",
    "monochrome graphite with subtle skin-tone warmth",
)

MOODS = (
    "quietly determined",
    "curious and observant",
    "elegant but unguarded",
    "tense anticipation",
    "calm self-possession",
    "playful confidence",
    "reflective solitude",
    "controlled theatrical drama",
)

CAMERAS = {
    "Portrait": (
        "full-body vertical editorial framing from a slightly low viewpoint",
        "layered vertical composition with foreground depth and a complete silhouette",
        "environmental portrait framing with strong architectural leading lines",
    ),
    "Landscape": (
        "wide environmental storytelling with the subject placed off-center",
        "cinematic lateral composition with foreground, subject, and distant depth",
        "low wide viewpoint emphasizing scale and location",
    ),
    "Square": (
        "balanced square composition with controlled asymmetry",
        "center-weighted environmental composition with geometric negative space",
        "square editorial framing built around repeating shapes",
    ),
}

LEGACY_CATEGORIES: Mapping[str, tuple[str, ...]] = {
    "bedroom interiors": ("bedroom", "bed ", "silk sheet"),
    "luxury penthouses": ("penthouse", "luxury suite", "hotel suite"),
    "reclining or kneeling poses": ("reclining", "kneeling", "lying on"),
    "warm golden lighting": ("golden hour", "warm window", "warm studio", "warm ambient"),
    "rainy neon streets": ("neon-lit", "tokyo street", "rain-washed", "reflective pavement"),
    "dark studio backdrops": ("dark backdrop", "photography studio", "studio light"),
}

FEEDBACK_GUIDANCE = {
    "prompt_match": "literal subject and action adherence",
    "eyes": "symmetrical eyes, aligned pupils, a natural gaze, and clear iris detail",
    "anatomy": "anatomically coherent poses, natural hands, and stable facial features",
    "composition": "intentional composition with clean subject framing",
    "style": "a strong and internally consistent visual style",
    "detail": "clear facial features and coherent fine detail",
    "content_rating": "an unambiguous match to the requested content rating",
}

FEEDBACK_AXES = {
    "prompt_match": ("theme", "environment", "action"),
    "eyes": ("action", "camera", "lighting"),
    "anatomy": ("action", "camera", "lighting"),
    "composition": ("action", "camera"),
    "style": ("lighting", "palette", "mood"),
    "detail": ("lighting", "camera"),
    "content_rating": ("theme", "mood"),
}

_IGNORED_TOKENS = {
    "score", "source", "rating", "woman", "adult", "solo", "female", "focus",
    "quality", "detailed", "detail", "image", "scene", "style", "shot", "body",
    "realistic", "photorealistic", "photography", "professional", "composition",
    "lens", "mm", "skin", "texture", "natural", "clear", "high", "best", "uhd",
    "clothed", "person", "subject",
    "lifelike", "facial", "material", "response", "accurate", "lighting", "framing",
    "refined", "crisp", "focal", "complete", "full-body", "clearly", "only",
}


def _tokens(value: str) -> set[str]:
    """Return semantic prompt tokens while excluding deterministic wrapper noise."""
    return {
        token
        for token in re.findall(r"[a-z][a-z-]{2,}", value.lower())
        if token not in _IGNORED_TOKENS and not token.startswith(("score_", "rating_", "source_"))
    }


def prompt_similarity(left: str, right: str) -> float:
    """Score semantic token overlap from 0 to 1 without external ML dependencies."""
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    jaccard = intersection / len(left_tokens | right_tokens)
    containment = intersection / min(len(left_tokens), len(right_tokens))
    return round((0.55 * containment) + (0.45 * jaccard), 4)


def maximum_similarity(prompt: str, recent_prompts: Iterable[str]) -> float:
    """Return the greatest similarity to recent prompts, or zero for no history."""
    return max((prompt_similarity(prompt, prior) for prior in recent_prompts), default=0.0)


def _parse_signature(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return {str(key): str(item) for key, item in parsed.items()} if isinstance(parsed, dict) else {}


def _recent_avoidances(
    history: Iterable[PromptHistoryItem],
) -> tuple[set[str], list[tuple[dict[str, str], int | None, tuple[str, ...]]]]:
    labels: set[str] = set()
    signatures: list[tuple[dict[str, str], int | None, tuple[str, ...]]] = []
    for item in history:
        signature = _parse_signature(item.novelty_signature)
        if signature:
            signatures.append((signature, item.user_rating, item.feedback_reasons))
            labels.update(
                value for key, value in signature.items() if key in {"theme", "environment", "action", "lighting"}
            )
        lowered = item.positive_prompt.lower()
        labels.update(label for label, needles in LEGACY_CATEGORIES.items() if any(needle in lowered for needle in needles))
    return labels, signatures


def _candidate_penalty(
    brief: NoveltyBrief,
    signatures: list[tuple[dict[str, str], int | None, tuple[str, ...]]],
) -> float:
    """Penalize recently repeated axes, weighting the newest records most heavily."""
    fields = ("theme", "environment", "action", "lighting", "palette", "mood", "camera")
    penalty = 0.0
    for index, (signature, rating, reasons) in enumerate(signatures[:20]):
        recency = 1.0 / (1.0 + index * 0.22)
        matches = sum(signature.get(field) == getattr(brief, field) for field in fields)
        penalty += recency * matches
        # Preserve successful visual treatment lightly, but never overpower the
        # stronger penalty against repeating a complete scene or environment.
        relevant_axes = {
            axis for reason in reasons for axis in FEEDBACK_AXES.get(reason, ())
        } or {"lighting", "palette", "mood", "camera"}
        treatment_matches = sum(
            signature.get(field) == getattr(brief, field) for field in relevant_axes
        )
        if rating is not None and rating >= 4:
            penalty -= recency * treatment_matches * 0.16
        elif rating is not None and rating <= 2:
            penalty += recency * treatment_matches * 0.24
    return penalty


def _quality_guidance(history: tuple[PromptHistoryItem, ...]) -> tuple[str, ...]:
    """Turn recent granular feedback into bounded prompt-level guidance."""
    issue_counts: dict[str, int] = {}
    strength_counts: dict[str, int] = {}
    for item in history[:20]:
        if item.user_rating is None or item.user_rating == 3:
            continue
        target = issue_counts if item.user_rating <= 2 else strength_counts
        for reason in item.feedback_reasons:
            if reason in FEEDBACK_GUIDANCE:
                target[reason] = target.get(reason, 0) + 1
    ranked = sorted(
        issue_counts,
        key=lambda reason: (issue_counts[reason], strength_counts.get(reason, 0)),
        reverse=True,
    )
    if not ranked:
        ranked = sorted(strength_counts, key=strength_counts.get, reverse=True)
    return tuple(FEEDBACK_GUIDANCE[reason] for reason in ranked[:2])


def _legacy_penalty(brief: NoveltyBrief, avoidances: set[str]) -> float:
    """Keep pre-migration prompt patterns from immediately reappearing."""
    candidate = " ".join(asdict(brief).values()).lower()
    penalty = 0.0
    for label, needles in LEGACY_CATEGORIES.items():
        if label in avoidances and any(needle in candidate for needle in needles):
            penalty += 3.0
    return penalty


def create_novelty_context(
    history: Iterable[PromptHistoryItem],
    *,
    creativity_level: str,
    aspect_ratio: str,
    content_rating: str = "Safe",
    rng: random.Random | None = None,
) -> NoveltyContext:
    """Choose a fresh coherent brief while using only local categorical history."""
    if creativity_level not in CREATIVITY_LEVELS:
        raise ValueError("Unsupported creativity level.")
    source = rng or random.SystemRandom()
    recent = tuple(history)
    avoidances, signatures = _recent_avoidances(recent)
    orientation = aspect_ratio.partition(" ")[0]
    cameras = CAMERAS.get(orientation, CAMERAS["Portrait"])
    if content_rating != "Safe":
        raise ValueError("NyxForge public edition supports Safe content only.")
    directions = SCENE_DIRECTIONS
    candidates: list[NoveltyBrief] = []
    for _ in range(48):
        direction = source.choice(directions)
        candidates.append(
            NoveltyBrief(
                theme=direction.theme,
                environment=source.choice(direction.environments),
                action=source.choice(direction.actions),
                lighting=source.choice(LIGHTING),
                palette=source.choice(PALETTES),
                mood=source.choice(MOODS),
                camera=source.choice(cameras),
            )
        )
    ranked = sorted(
        candidates,
        key=lambda candidate: _candidate_penalty(candidate, signatures)
        + _legacy_penalty(candidate, avoidances),
    )
    pool_sizes = {"consistent": 12, "balanced": 6, "experimental": 2}
    pool = ranked[: pool_sizes[creativity_level]]
    brief = source.choice(pool)
    limits = {"consistent": 4, "balanced": 6, "experimental": 8}
    avoid_summary = "; ".join(sorted(avoidances)[: limits[creativity_level]])
    thresholds = {"consistent": 0.46, "balanced": 0.37, "experimental": 0.30}
    return NoveltyContext(
        brief=brief,
        creativity_level=creativity_level,
        avoid_summary=avoid_summary,
        quality_guidance=_quality_guidance(recent),
        recent_prompts=tuple(item.positive_prompt for item in recent[:20]),
        similarity_threshold=thresholds[creativity_level],
    )


def provider_novelty_instruction(context: NoveltyContext, *, retry: bool = False) -> str:
    """Return a compact brief that never sends raw historical prompts to providers."""
    instruction = (
        "Follow this creative brief closely while keeping the result coherent: "
        f"{context.brief.instruction()}"
    )
    if context.avoid_summary:
        instruction += f" Do not reuse these recently overused ideas: {context.avoid_summary}."
    if context.quality_guidance:
        instruction += (
            " Prioritize these preferences learned from recent ratings: "
            f"{'; '.join(context.quality_guidance)}."
        )
    if retry:
        instruction += " The previous proposal resembled recent work; change the concrete staging and visual vocabulary substantially."
    return instruction
