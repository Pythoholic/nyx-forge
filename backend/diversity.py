"""Durable, bounded concept-level diversity memory for the prompt engine."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping

from . import database


def canonical_concept_ids(concept_ids: Iterable[str]) -> tuple[str, ...]:
    """Canonicalize a semantic recipe independently of rendering or word order."""
    return tuple(sorted(set(concept_ids)))


def concept_fingerprint(concept_ids: Iterable[str]) -> str:
    payload = "\x1f".join(canonical_concept_ids(concept_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


@dataclass(frozen=True)
class DiversityConfig:
    cooldown_window: int = 100
    similarity_threshold: float = .72
    minimum_weight: float = .08
    history_limit: int = 100
    max_fingerprints: int = 500
    max_usage_rows: int = 2_000
    max_cooccurrence_rows: int = 50_000


@dataclass(frozen=True)
class DiversitySnapshot:
    sequence: int
    recent_sets: tuple[frozenset[str], ...]
    concept_usage: Mapping[str, tuple[int, int]]
    cooccurrence: Mapping[tuple[str, ...], tuple[int, int]]
    config: DiversityConfig

    def recency_weight(self, concept_id: str) -> float:
        usage = self.concept_usage.get(concept_id)
        if usage is None:
            return 1.0
        count, last_sequence = usage
        age = max(0, self.sequence - last_sequence)
        recovery = min(1.0, age / max(self.config.cooldown_window, 1))
        frequency = 1.0 / (1.0 + .18 * max(count - 1, 0))
        penalized = self.config.minimum_weight + (1 - self.config.minimum_weight) * recovery
        return max(self.config.minimum_weight, min(1.0, penalized * frequency))

    def combination_weight(self, selected: Iterable[str], candidate: str) -> float:
        ids = canonical_concept_ids((*selected, candidate))
        penalties: list[float] = []
        for arity in (2, 3):
            for combo in combinations(ids, arity):
                tracked = self.cooccurrence.get(combo)
                if tracked is not None:
                    count, last_sequence = tracked
                    age = max(0, self.sequence - last_sequence)
                    freshness = max(0.0, 1 - age / max(self.config.cooldown_window, 1))
                    penalties.append(freshness * min(1.0, count / 4))
        return max(self.config.minimum_weight, 1 - .55 * max(penalties, default=0.0))

    def maximum_similarity(self, concept_ids: Iterable[str]) -> float:
        ids = frozenset(concept_ids)
        return max((jaccard_similarity(ids, old) for old in self.recent_sets), default=0.0)

    def uniqueness_score(self, concept_ids: Iterable[str]) -> float:
        ids = canonical_concept_ids(concept_ids)
        distance = 1 - self.maximum_similarity(ids)
        repeats: list[float] = []
        for arity in (2, 3):
            for combo in combinations(ids, arity):
                tracked = self.cooccurrence.get(combo)
                if tracked:
                    count, last_sequence = tracked
                    age = max(0, self.sequence - last_sequence)
                    repeats.append(min(1.0, count / 5) * max(
                        0.0, 1 - age / max(self.config.cooldown_window, 1)))
        return max(0.0, min(1.0, .75 * distance + .25 * (1 - max(repeats, default=0.0))))


class DiversityStore:
    def __init__(self, config: DiversityConfig | None = None):
        self.config = config or DiversityConfig()

    def load(self) -> DiversitySnapshot:
        with database._connection() as connection:
            database.ensure_diversity_tables(connection)
            latest = connection.execute(
                "SELECT COALESCE(MAX(id), 0) FROM diversity_fingerprints"
            ).fetchone()[0]
            rows = connection.execute(
                "SELECT concept_ids FROM diversity_fingerprints ORDER BY id DESC LIMIT ?",
                (min(self.config.history_limit, self.config.max_fingerprints),),
            ).fetchall()
            usage = connection.execute(
                "SELECT concept_id, use_count, last_sequence FROM diversity_concept_usage"
            ).fetchall()
            cooccurrence = connection.execute(
                "SELECT concept_key, use_count, last_sequence FROM diversity_cooccurrence"
            ).fetchall()
        return DiversitySnapshot(
            int(latest), tuple(frozenset(json.loads(row[0])) for row in rows),
            {str(row[0]): (int(row[1]), int(row[2])) for row in usage},
            {tuple(str(row[0]).split("\x1f")): (int(row[1]), int(row[2]))
             for row in cooccurrence}, self.config,
        )

    def record(self, concept_ids: Iterable[str], model: str, orientation: str) -> str:
        ids = canonical_concept_ids(concept_ids)
        fingerprint = concept_fingerprint(ids)
        with database._connection() as connection:
            database.ensure_diversity_tables(connection)
            cursor = connection.execute(
                "INSERT INTO diversity_fingerprints "
                "(fingerprint, concept_ids, model, orientation) VALUES (?, ?, ?, ?)",
                (fingerprint, json.dumps(ids, separators=(",", ":")), model, orientation),
            )
            sequence = int(cursor.lastrowid)
            for concept_id in ids:
                connection.execute(
                    "INSERT INTO diversity_concept_usage VALUES (?, 1, ?) "
                    "ON CONFLICT(concept_id) DO UPDATE SET "
                    "use_count=use_count+1, last_sequence=excluded.last_sequence",
                    (concept_id, sequence),
                )
            for arity in (2, 3):
                for combo in combinations(ids, arity):
                    key = "\x1f".join(combo)
                    connection.execute(
                        "INSERT INTO diversity_cooccurrence VALUES (?, ?, 1, ?) "
                        "ON CONFLICT(concept_key, arity) DO UPDATE SET "
                        "use_count=use_count+1, last_sequence=excluded.last_sequence",
                        (key, arity, sequence),
                    )
            self._prune(connection, sequence)
        return fingerprint

    def _prune(self, connection, sequence: int) -> None:
        cutoff = max(0, sequence - self.config.cooldown_window)
        connection.execute(
            "DELETE FROM diversity_fingerprints WHERE id <= ? OR id NOT IN "
            "(SELECT id FROM diversity_fingerprints ORDER BY id DESC LIMIT ?)",
            (max(0, sequence - self.config.max_fingerprints), self.config.max_fingerprints),
        )
        connection.execute(
            "DELETE FROM diversity_concept_usage WHERE last_sequence <= ?", (cutoff,))
        connection.execute(
            "DELETE FROM diversity_cooccurrence WHERE last_sequence <= ?", (cutoff,))
        for table, cap, order in (
            ("diversity_concept_usage", self.config.max_usage_rows, "last_sequence"),
            ("diversity_cooccurrence", self.config.max_cooccurrence_rows, "last_sequence"),
        ):
            connection.execute(
                f"DELETE FROM {table} WHERE rowid NOT IN "
                f"(SELECT rowid FROM {table} ORDER BY {order} DESC, rowid DESC LIMIT ?)",
                (cap,),
            )
