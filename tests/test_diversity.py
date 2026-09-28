import sqlite3
from collections import Counter
from unittest.mock import patch

from backend import database
from backend.diversity import (
    DiversityConfig, DiversityStore, canonical_concept_ids, concept_fingerprint,
)
from backend.prompt_engine import JuggernautRagnarokAdapter, MODEL_PROFILES, PromptEngine


def _store(tmp_path, **overrides):
    config = DiversityConfig(**overrides)
    path = tmp_path / "diversity.db"
    patcher = patch.object(database, "DATABASE_PATH", path)
    patcher.start()
    database.initialize_database()
    return DiversityStore(config), patcher, path


def _ids(result):
    return canonical_concept_ids(c.id for c in result.positive_fragments
                                 if c.category != "control")


def test_concept_fingerprint_is_stable_and_text_independent():
    assert concept_fingerprint(["scene.a", "subject.b"]) == concept_fingerprint(
        ["subject.b", "scene.a", "scene.a"])
    assert concept_fingerprint(["scene.a", "subject.b"]) != concept_fingerprint(
        ["scene.a", "subject.c"])


def test_recent_concept_is_weighted_down_but_never_eliminated(tmp_path):
    store, patcher, _ = _store(tmp_path, cooldown_window=10, minimum_weight=.07)
    try:
        store.record(["subject.a", "scene.a"], "m", "front")
        snapshot = store.load()
        assert 0 < snapshot.recency_weight("subject.a") < snapshot.recency_weight("new")
        assert snapshot.recency_weight("subject.a") >= .07
    finally:
        patcher.stop()


def test_sequential_generation_improves_short_window_concept_diversity(tmp_path):
    store, patcher, _ = _store(tmp_path, cooldown_window=60)
    try:
        enabled = PromptEngine(diversity_store=store, diversity_enabled=True)
        baseline = PromptEngine(diversity_enabled=False)
        diverse_results = [enabled.generate("juggernaut_ragnarok", i) for i in range(60)]
        baseline_results = [baseline.generate("juggernaut_ragnarok", i) for i in range(60)]
        # Metric: mean unique concepts per rolling ten-prompt window.
        def metric(results):
            windows = []
            sets = [set(_ids(result)) for result in results]
            for start in range(len(sets) - 9):
                windows.append(len(set().union(*sets[start:start + 10])))
            return sum(windows) / len(windows)
        assert metric(diverse_results) > metric(baseline_results)
    finally:
        patcher.stop()


def test_similarity_rejection_avoids_deliberately_repeated_candidate(tmp_path):
    store, patcher, _ = _store(tmp_path, similarity_threshold=.5)
    try:
        baseline = PromptEngine(diversity_enabled=False)
        repeated = baseline.generate("juggernaut_ragnarok", 1)
        store.record(_ids(repeated), repeated.ast.model_id, repeated.ast.orientation.value)
        engine = PromptEngine(diversity_store=store, diversity_enabled=True)
        result = engine.generate("juggernaut_ragnarok", 1)
        assert _ids(result) != _ids(repeated)
        assert store.load().recent_sets[0] == frozenset(_ids(result))
    finally:
        patcher.stop()


def test_all_similar_candidates_return_least_similar_instead_of_raising(tmp_path):
    store, patcher, _ = _store(tmp_path, similarity_threshold=0.0)
    try:
        engine = PromptEngine(diversity_store=store, diversity_enabled=True)
        first = engine.generate("juggernaut_ragnarok", 4)
        snapshot = store.load()
        candidates = engine.generate_candidates(
            "juggernaut_ragnarok", 4, first.ast.orientation, first.ast.rating, snapshot)
        profile = MODEL_PROFILES["juggernaut_ragnarok"]
        adapter = JuggernautRagnarokAdapter(engine.tokenizer, profile)
        visible_candidates = []
        for candidate in candidates:
            _, _, positive, _, _, _ = engine._compile_candidate(
                candidate, profile, adapter,
            )
            visible_candidates.append(
                tuple(c.id for c in positive if c.category != "control")
            )
        minimum = min(map(snapshot.maximum_similarity, visible_candidates))
        second = engine.generate("juggernaut_ragnarok", 4)
        assert second.positive
        assert snapshot.maximum_similarity(_ids(second)) == minimum
    finally:
        patcher.stop()


def test_state_persists_across_fresh_store_loader(tmp_path):
    store, patcher, path = _store(tmp_path)
    try:
        store.record(["a", "b", "c"], "m", "side")
        fresh = DiversityStore(store.config).load()
        assert fresh.recent_sets[0] == frozenset({"a", "b", "c"})
        assert fresh.concept_usage["a"][0] == 1
        assert ("a", "b") in fresh.cooccurrence
        assert path.exists()
    finally:
        patcher.stop()


def test_pruning_and_eviction_bound_all_tables(tmp_path):
    store, patcher, _ = _store(
        tmp_path, cooldown_window=3, max_fingerprints=4,
        max_usage_rows=5, max_cooccurrence_rows=7,
    )
    try:
        for index in range(12):
            store.record([f"a{index}", f"b{index}", f"c{index}"], "m", "front")
        with database._connection() as connection:
            assert connection.execute("SELECT COUNT(*) FROM diversity_fingerprints").fetchone()[0] <= 4
            assert connection.execute("SELECT COUNT(*) FROM diversity_concept_usage").fetchone()[0] <= 5
            assert connection.execute("SELECT COUNT(*) FROM diversity_cooccurrence").fetchone()[0] <= 7
    finally:
        patcher.stop()


def test_existing_database_migrates_without_loss(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE prompts (id INTEGER PRIMARY KEY, positive_prompt TEXT NOT NULL, negative_prompt TEXT NOT NULL, timestamp TEXT NOT NULL)")
        connection.execute("INSERT INTO prompts VALUES (1, 'keep me', 'negative', '2025-01-01')")
    with patch.object(database, "DATABASE_PATH", path):
        database.initialize_database()
        database.initialize_database()
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT positive_prompt FROM prompts WHERE id=1").fetchone()[0] == "keep me"
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"diversity_fingerprints", "diversity_concept_usage", "diversity_cooccurrence"} <= tables
