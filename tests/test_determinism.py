"""Cross-scenario determinism — the gym's core reproducibility guarantee.

"Same seed -> same scenario -> same grader results" only holds if generation is
free of wall-clock and other non-seeded entropy. Regression guard: this caught
`_base_time = datetime.now()`, which made heavy scenarios (queue_management)
differ between two generations from the same seed whenever they straddled a
second boundary. Covers every registered scenario, not just phishing.
"""

import pytest

from scenarios import SCENARIO_REGISTRY


@pytest.mark.parametrize("name", sorted(SCENARIO_REGISTRY))
def test_same_seed_is_byte_identical(name):
    cls = SCENARIO_REGISTRY[name]
    first = cls(seed=42).generate().model_dump_json()
    second = cls(seed=42).generate().model_dump_json()
    assert first == second, f"{name} is non-deterministic for a fixed seed"


@pytest.mark.parametrize("name", sorted(SCENARIO_REGISTRY))
def test_different_seeds_differ(name):
    cls = SCENARIO_REGISTRY[name]
    a = cls(seed=42).generate().model_dump_json()
    b = cls(seed=43).generate().model_dump_json()
    assert a != b, f"{name} ignores its seed"


def test_base_time_is_fixed_not_wallclock():
    from scenarios.base import EPISODE_BASE_TIME, BaseScenario

    class _Probe(BaseScenario):
        def generate(self):  # pragma: no cover - not called
            raise NotImplementedError

    # Two instances created at (slightly) different wall-clock times must share
    # the same base time — otherwise timestamps would drift between generations.
    assert _Probe(seed=1)._base_time == EPISODE_BASE_TIME
    assert _Probe(seed=2)._base_time == EPISODE_BASE_TIME
