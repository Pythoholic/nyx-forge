"""The lightweight test wiring for the 10,000-prompt preflight gate."""
from scripts.preflight import DISTRIBUTION_NAMES, ERROR_NAMES, run_preflight


def test_application_path_preflight_has_zero_hardening_errors():
    report = run_preflight(60)
    assert report["generated"] == 60
    assert report["errors"] == {name: 0 for name in ERROR_NAMES}
    assert report["token_maxima"]["positive"] <= 75
    assert report["token_maxima"]["negative"] <= 75
    assert "environment_action_conflict" in report["errors"]
    assert set(report["distribution"]) == set(DISTRIBUTION_NAMES)
    # Every family must be reachable during selection. Optional Tier-3 details
    # may all be pruned from a small smoke sample at the 75-token hard cap.
    assert all(report["distribution"][name]["selected_prompts"] > 0
               for name in DISTRIBUTION_NAMES)
    for name in (
        "skin", "build_core", "wardrobe", "pose",
        "environment", "lighting", "primary_medium",
    ):
        assert report["distribution"][name]["unique_records"] > 0
    assert report["distribution"]["build_core"]["selected_prompts"] == 60
    assert report["distribution"]["build_core"]["emitted_prompts"] == 60
    assert set(report["candidate_survival"]["valid_candidate_histogram"]) == {
        str(value) for value in range(1, 9)
    }
    assert report["candidate_survival"]["valid_candidate_histogram"]["8"] == 60
    assert report["candidate_survival"]["valid_candidates"] == 60 * 8
    assert report["candidate_survival"]["candidate_attempts"] == (
        report["candidate_survival"]["valid_candidates"] +
        report["candidate_survival"]["rejected_candidates"]
    )
    for model in ("cyberrealistic_pony", "juggernaut_ragnarok"):
        for family in ("primary_medium", "finish"):
            reachability = report["reachability"][model][family]
            assert reachability["pool_size"] == reachability["eligible"]
            assert reachability["selected"] > 0
            if model == "juggernaut_ragnarok":
                assert reachability["emitted"] > 0
