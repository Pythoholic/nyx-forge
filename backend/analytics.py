"""Pure aggregation helpers for the local generation dashboard."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from math import ceil
import re
from statistics import mean, median
from typing import Iterable

from .forge import FALLBACK_MODEL_PROFILE, model_profile_for
from .versioning import PROMPT_COMPILER_VERSION


@dataclass(frozen=True, slots=True)
class AnalyticsRun:
    """One completed generation with its reproducibility and outcome fields."""

    id: int
    timestamp: str
    model: str
    style: str
    content_rating: str
    aspect_ratio: str
    quality_mode: str
    width: int
    height: int
    sampler_name: str
    scheduler: str
    steps: int
    cfg_scale: float
    duration_seconds: float | None
    diffusion_duration_seconds: float | None
    upscale_duration_seconds: float | None
    prompt_duration_seconds: float | None
    prompt_engine: str
    creativity_level: str
    similarity_score: float | None
    novelty_retry_count: int
    user_rating: int | None
    feedback_reasons: tuple[str, ...] = ()
    compiler_version: str = PROMPT_COMPILER_VERSION

    @property
    def megapixels(self) -> float:
        """Return final output size in megapixels."""
        return self.width * self.height / 1_000_000


@dataclass(frozen=True, slots=True)
class AttemptCounts:
    """Terminal prompt/image workflow outcomes for reliability reporting."""

    succeeded: int = 0
    failed: int = 0
    failure_reasons: tuple[tuple[str, int], ...] = ()


def canonical_model_name(value: str) -> str:
    """Collapse Forge hashes and duplicate filenames into one checkpoint label."""
    profile = model_profile_for(value)
    if profile is not FALLBACK_MODEL_PROFILE:
        return profile.display_name
    filename = value.replace("/", "\\").rsplit("\\", 1)[-1]
    filename = filename.removesuffix(".safetensors")
    filename = re.sub(
        r"\s*\[[0-9a-f]{8,64}\]\s*$", "", filename, flags=re.IGNORECASE
    )
    filename = re.sub(r"\s*\(\d+\)\s*$", "", filename)
    return filename.replace("_", " ").strip() or "Unknown model"


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    """Return a nearest-rank percentile, which remains clear for small samples."""
    ordered = sorted(value for value in values if value >= 0)
    if not ordered:
        return None
    rank = max(0, ceil(percentile * len(ordered)) - 1)
    return round(ordered[rank], 2)


def _round_or_none(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None


def _group_summary(label: str, runs: list[AnalyticsRun]) -> dict[str, object]:
    durations = [item.duration_seconds for item in runs if item.duration_seconds is not None]
    prompt_durations = [
        item.prompt_duration_seconds
        for item in runs
        if item.prompt_duration_seconds is not None
    ]
    diffusion_durations = [
        item.diffusion_duration_seconds
        for item in runs
        if item.diffusion_duration_seconds is not None
    ]
    upscale_durations = [
        item.upscale_duration_seconds
        for item in runs
        if item.upscale_duration_seconds is not None
    ]
    ratings = [item.user_rating for item in runs if item.user_rating is not None]
    similarity_scores = [
        item.similarity_score
        for item in runs
        if item.similarity_score is not None
    ]
    megapixels = [item.megapixels for item in runs]
    seconds_per_megapixel = [
        item.duration_seconds / item.megapixels
        for item in runs
        if item.duration_seconds is not None and item.megapixels > 0
    ]
    return {
        "label": label,
        "count": len(runs),
        "average_seconds": _round_or_none(mean(durations) if durations else None),
        "median_seconds": _round_or_none(median(durations) if durations else None),
        "p90_seconds": _percentile(durations, 0.9),
        "average_diffusion_seconds": _round_or_none(
            mean(diffusion_durations) if diffusion_durations else None
        ),
        "average_upscale_seconds": _round_or_none(
            mean(upscale_durations) if upscale_durations else None
        ),
        "average_prompt_seconds": _round_or_none(
            mean(prompt_durations) if prompt_durations else None
        ),
        "prompt_timing_count": len(prompt_durations),
        "average_rating": _round_or_none(mean(ratings) if ratings else None),
        "rated_count": len(ratings),
        "average_similarity": _round_or_none(
            mean(similarity_scores) if similarity_scores else None
        ),
        "similarity_count": len(similarity_scores),
        "duplicate_risk_count": sum(score >= 0.46 for score in similarity_scores),
        "average_megapixels": _round_or_none(mean(megapixels) if megapixels else None),
        "seconds_per_megapixel": _round_or_none(
            mean(seconds_per_megapixel) if seconds_per_megapixel else None
        ),
    }


def _breakdown(
    runs: list[AnalyticsRun], key
) -> list[dict[str, object]]:  # type: ignore[no-untyped-def]
    grouped: dict[str, list[AnalyticsRun]] = defaultdict(list)
    for item in runs:
        grouped[str(key(item))].append(item)
    return sorted(
        (_group_summary(label, values) for label, values in grouped.items()),
        key=lambda item: (-int(item["count"]), str(item["label"])),
    )


def _evidence_confidence(sample_count: int) -> str:
    """Translate local sample size into an honest evidence label."""
    if sample_count >= 10:
        return "high"
    if sample_count >= 5:
        return "medium"
    return "learning"


def _build_recommendations(
    completed: list[AnalyticsRun],
    combinations: list[dict[str, object]],
    by_prompt_engine: list[dict[str, object]],
    attempts: AttemptCounts,
) -> list[dict[str, object]]:
    """Turn observed local outcomes into cautious, actionable suggestions."""
    ratings = [item.user_rating for item in completed if item.user_rating is not None]
    prior_rating = mean(ratings) if ratings else 3.0
    recommendations: list[dict[str, object]] = []

    quality_candidates = [
        item
        for item in combinations
        if int(item["rated_count"]) >= 2 and item["average_rating"] is not None
    ]
    if quality_candidates:
        def adjusted_rating(item: dict[str, object]) -> float:
            rated_count = int(item["rated_count"])
            observed = float(item["average_rating"])
            # A light Bayesian shrinkage prevents a one-off 5-star result from
            # defeating a well-tested 4-star recipe.
            return (observed * rated_count + prior_rating * 4) / (rated_count + 4)

        winner = max(
            quality_candidates,
            key=lambda item: (adjusted_rating(item), int(item["rated_count"])),
        )
        recommendations.append(
            {
                "id": "quality-recipe",
                "category": "quality",
                "title": "Best observed quality recipe",
                "summary": (
                    f"{winner['style']} at {winner['quality_mode']} quality is your "
                    f"strongest rated repeatable combination for this checkpoint."
                ),
                "action": (
                    f"Start with {winner['aspect_ratio']}, {winner['sampler_name']} / "
                    f"{winner['scheduler']}, {winner['steps']} steps, CFG {winner['cfg_scale']}."
                ),
                "confidence": _evidence_confidence(int(winner["rated_count"])),
                "evidence": [
                    f"{winner['average_rating']} / 5 average",
                    f"{winner['rated_count']} rated of {winner['count']} runs",
                    (
                        f"{winner['median_seconds']}s median render"
                        if winner["median_seconds"] is not None
                        else "Render timing not yet measured"
                    ),
                ],
                "settings": {
                    "model": winner["model"],
                    "style": winner["style"],
                    "quality_mode": winner["quality_mode"],
                    "aspect_ratio": winner["aspect_ratio"],
                    "prompt_engine": winner["prompt_engine"],
                },
            }
        )

    speed_candidates = [
        item
        for item in combinations
        if int(item["count"]) >= 2
        and item["seconds_per_megapixel"] is not None
        and (
            item["average_rating"] is None
            or float(item["average_rating"]) >= prior_rating - 0.35
        )
    ]
    if speed_candidates:
        fastest = min(
            speed_candidates,
            key=lambda item: (
                float(item["seconds_per_megapixel"]),
                float(item["median_seconds"] or float("inf")),
            ),
        )
        recommendations.append(
            {
                "id": "speed-recipe",
                "category": "speed",
                "title": "Fastest quality-conscious recipe",
                "summary": (
                    f"{fastest['quality_mode'].title()} quality with {fastest['style']} "
                    "currently gives the best measured throughput without a clear rating penalty."
                ),
                "action": (
                    f"Use {fastest['aspect_ratio']} with {fastest['steps']} steps when speed matters; "
                    "move up a quality tier only for final selects."
                ),
                "confidence": _evidence_confidence(int(fastest["count"])),
                "evidence": [
                    f"{fastest['seconds_per_megapixel']}s per megapixel",
                    f"{fastest['count']} measured runs",
                    (
                        f"{fastest['average_rating']} / 5 average"
                        if fastest["average_rating"] is not None
                        else "Needs more ratings"
                    ),
                ],
                "settings": {
                    "model": fastest["model"],
                    "style": fastest["style"],
                    "quality_mode": fastest["quality_mode"],
                    "aspect_ratio": fastest["aspect_ratio"],
                },
            }
        )

    engine_candidates = [
        item
        for item in by_prompt_engine
        if str(item["label"]) != "unknown"
        and int(item["rated_count"]) >= 2
        and item["average_rating"] is not None
    ]
    if engine_candidates:
        best_engine = max(
            engine_candidates,
            key=lambda item: (
                (float(item["average_rating"]) * int(item["rated_count"]) + prior_rating * 4)
                / (int(item["rated_count"]) + 4),
                int(item["rated_count"]),
            ),
        )
        recommendations.append(
            {
                "id": "prompt-engine",
                "category": "prompt",
                "title": "Prompt engine to favor",
                "summary": (
                    f"{best_engine['label']} has produced the strongest rated images in your "
                    "local history after accounting for sample size."
                ),
                "action": (
                    "Use it as the default for important generations, while keeping Explore "
                    "enabled when you want more scene variety."
                ),
                "confidence": _evidence_confidence(int(best_engine["rated_count"])),
                "evidence": [
                    f"{best_engine['average_rating']} / 5 average",
                    f"{best_engine['rated_count']} rated of {best_engine['count']} generations",
                    (
                        f"{best_engine['average_prompt_seconds']}s average prompt time"
                        if best_engine["average_prompt_seconds"] is not None
                        else "Prompt timing coverage is still growing"
                    ),
                ],
                "settings": {"prompt_engine": best_engine["label"]},
            }
        )

    total = len(completed)
    rated_count = len(ratings)
    rating_coverage = rated_count / total * 100 if total else 0
    if rating_coverage < 70:
        recommendations.append(
            {
                "id": "rating-coverage",
                "category": "data",
                "title": "Rate more results for sharper guidance",
                "summary": (
                    f"Only {rating_coverage:.0f}% of generated images have feedback, so prompt "
                    "and quality rankings still contain blind spots."
                ),
                "action": "Rate both strong and weak images; the contrast is what teaches the novelty engine.",
                "confidence": "high",
                "evidence": [
                    f"{rated_count} rated images",
                    f"{total - rated_count} unrated images",
                    "Target at least 70% coverage",
                ],
                "settings": {},
            }
        )

    terminal_attempts = attempts.succeeded + attempts.failed
    if terminal_attempts and attempts.failed / terminal_attempts >= 0.08:
        failure_rate = attempts.failed / terminal_attempts * 100
        recommendations.append(
            {
                "id": "reliability",
                "category": "reliability",
                "title": "Protect iteration speed from failed runs",
                "summary": f"{failure_rate:.0f}% of terminal attempts failed in this window.",
                "action": (
                    "Prototype in Normal mode first, then promote successful compositions to High "
                    "or Super; this avoids spending VRAM on an unproven scene."
                ),
                "confidence": _evidence_confidence(terminal_attempts),
                "evidence": [
                    f"{attempts.failed} failed attempts",
                    f"{attempts.succeeded} successful attempts",
                    (
                        f"Most common: {attempts.failure_reasons[0][0]} "
                        f"({attempts.failure_reasons[0][1]})"
                        if attempts.failure_reasons
                        else "Normal-first preserves the same prompt for refinement"
                    ),
                ],
                "settings": {},
            }
        )

    return recommendations


def summarize_analytics(
    runs: Iterable[AnalyticsRun], attempts: AttemptCounts, *, days: int
) -> dict[str, object]:
    """Build one dashboard-ready snapshot without hiding incomplete coverage."""
    observed = list(runs)
    completed = [
        item for item in observed if item.compiler_version == PROMPT_COMPILER_VERSION
    ]
    legacy_generations_excluded = len(observed) - len(completed)
    durations = [item.duration_seconds for item in completed if item.duration_seconds is not None]
    prompt_durations = [
        item.prompt_duration_seconds
        for item in completed
        if item.prompt_duration_seconds is not None
    ]
    ratings = [item.user_rating for item in completed if item.user_rating is not None]
    phase_timing_count = sum(
        item.diffusion_duration_seconds is not None for item in completed
    )
    terminal_attempts = attempts.succeeded + attempts.failed

    daily_runs: dict[str, list[AnalyticsRun]] = defaultdict(list)
    for item in completed:
        try:
            day = datetime.fromisoformat(item.timestamp).date().isoformat()
        except ValueError:
            day = item.timestamp[:10]
        daily_runs[day].append(item)
    timeline = []
    for day, values in sorted(daily_runs.items()):
        day_durations = [
            item.duration_seconds for item in values if item.duration_seconds is not None
        ]
        day_ratings = [item.user_rating for item in values if item.user_rating is not None]
        timeline.append(
            {
                "date": day,
                "count": len(values),
                "average_seconds": _round_or_none(
                    mean(day_durations) if day_durations else None
                ),
                "average_rating": _round_or_none(
                    mean(day_ratings) if day_ratings else None
                ),
            }
        )

    combination_groups: dict[tuple[object, ...], list[AnalyticsRun]] = defaultdict(list)
    for item in completed:
        combination_groups[
            (
                canonical_model_name(item.model),
                item.style,
                item.content_rating,
                item.quality_mode,
                item.aspect_ratio,
                item.prompt_engine,
                item.creativity_level,
                item.sampler_name,
                item.scheduler,
                item.steps,
                item.cfg_scale,
            )
        ].append(item)
    combinations = []
    for key, values in combination_groups.items():
        summary = _group_summary(str(key[0]), values)
        combinations.append(
            {
                **summary,
                "model": key[0],
                "style": key[1],
                "content_rating": key[2],
                "quality_mode": key[3],
                "aspect_ratio": key[4],
                "prompt_engine": key[5],
                "creativity_level": key[6],
                "sampler_name": key[7],
                "scheduler": key[8],
                "steps": key[9],
                "cfg_scale": key[10],
            }
        )
    combinations.sort(
        key=lambda item: (
            -int(item["rated_count"]),
            -(float(item["average_rating"]) if item["average_rating"] is not None else -1),
            -int(item["count"]),
        )
    )

    by_model = _breakdown(completed, lambda item: canonical_model_name(item.model))
    by_quality = _breakdown(completed, lambda item: item.quality_mode)
    by_style = _breakdown(completed, lambda item: item.style)
    by_content_rating = _breakdown(completed, lambda item: item.content_rating)
    by_prompt_engine = _breakdown(completed, lambda item: item.prompt_engine)
    rating_distribution = Counter(ratings)
    low_score_issues = Counter(
        reason
        for item in completed
        if item.user_rating is not None and item.user_rating <= 2
        for reason in item.feedback_reasons
    )
    high_score_strengths = Counter(
        reason
        for item in completed
        if item.user_rating is not None and item.user_rating >= 4
        for reason in item.feedback_reasons
    )
    similarity_scores = [
        item.similarity_score
        for item in completed
        if item.similarity_score is not None
    ]
    return {
        "window_days": days,
        "summary": {
            "compiler_version": PROMPT_COMPILER_VERSION,
            "legacy_generations_excluded": legacy_generations_excluded,
            "total_generations": len(completed),
            "measured_generations": len(durations),
            "average_seconds": _round_or_none(mean(durations) if durations else None),
            "median_seconds": _round_or_none(median(durations) if durations else None),
            "p90_seconds": _percentile(durations, 0.9),
            "average_prompt_seconds": _round_or_none(
                mean(prompt_durations) if prompt_durations else None
            ),
            "prompt_timing_count": len(prompt_durations),
            "phase_timing_count": phase_timing_count,
            "average_rating": _round_or_none(mean(ratings) if ratings else None),
            "rated_count": len(ratings),
            "success_rate": (
                round(attempts.succeeded / terminal_attempts * 100, 1)
                if terminal_attempts
                else None
            ),
            "successful_attempts": attempts.succeeded,
            "failed_attempts": attempts.failed,
            "failure_reasons": [
                {"reason": reason, "count": count}
                for reason, count in attempts.failure_reasons
            ],
            "average_similarity": _round_or_none(
                mean(similarity_scores) if similarity_scores else None
            ),
            "similarity_count": len(similarity_scores),
            "duplicate_risk_count": sum(
                score >= 0.46 for score in similarity_scores
            ),
        },
        "timeline": timeline,
        "by_model": by_model,
        "by_quality": by_quality,
        "by_style": by_style,
        "by_content_rating": by_content_rating,
        "by_prompt_engine": by_prompt_engine,
        "rating_distribution": [
            {"score": score, "count": rating_distribution.get(score, 0)}
            for score in range(1, 6)
        ],
        "quality_signals": {
            "issues": [
                {"reason": reason, "count": count}
                for reason, count in low_score_issues.most_common()
            ],
            "strengths": [
                {"reason": reason, "count": count}
                for reason, count in high_score_strengths.most_common()
            ],
        },
        "top_combinations": combinations[:12],
        "recommendations": _build_recommendations(
            completed, combinations, by_prompt_engine, attempts
        ),
        "recent_runs": [
            {**asdict(item), "megapixels": round(item.megapixels, 2)}
            for item in completed[:20]
        ],
    }
