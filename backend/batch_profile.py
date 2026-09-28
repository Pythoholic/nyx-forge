"""Per-batch adjustments to which Safe keyword records the engine may select."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

@dataclass(frozen=True)
class BatchProfile:
    """A public extension point that may only narrow Safe keyword pools."""

    def wardrobe_pool(self, records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        """Return the wardrobe records this profile allows."""
        return list(records)

    def pose_weight(self, record: Mapping[str, Any]) -> float:
        """Return the multiplier for one pose record under this profile."""
        return 1.0


NEUTRAL = BatchProfile()

# The named profiles a batch runner can ask for by string, so a runner does not
# have to import this module or know the weights.
PROFILES: Mapping[str, BatchProfile] = {}


def profile_for(name: str | None) -> BatchProfile:
    """Return a named profile, or the neutral one for an unknown name."""
    return PROFILES.get(name or "", NEUTRAL)
