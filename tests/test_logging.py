"""Structured logging: JSON shape, request-ID propagation, episode events."""

import json
import logging

from server.logging_config import JsonFormatter, TextFormatter, _clean_request_id, configure_logging


def _record(msg="hello", **extra):
    record = logging.LogRecord("soc_gym.test", logging.INFO, __file__, 1, msg, None, None)
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_json_formatter_emits_one_object_with_extras():
    line = JsonFormatter().format(_record("episode.reset", task_id="phishing", seed=42, request_id="abc"))
    payload = json.loads(line)
    assert payload["msg"] == "episode.reset"
    assert payload["level"] == "INFO"
    assert payload["task_id"] == "phishing"
    assert payload["seed"] == 42
    assert payload["request_id"] == "abc"
    assert "\n" not in line


def test_text_formatter_appends_fields():
    out = TextFormatter().format(_record("x", task_id="phishing"))
    assert out.endswith("task_id=phishing")


def test_configure_logging_is_idempotent():
    root = logging.getLogger()
    configure_logging(fmt="json")
    configure_logging(fmt="json")
    ours = [h for h in root.handlers if getattr(h, "_soc_gym_handler", False)]
    assert len(ours) == 1


def test_request_id_is_sanitised():
    assert _clean_request_id("req-123") == "req-123"
    for bad in (None, "", "a" * 500, "evil\nINFO forged", "has space", "naïve"):
        cleaned = _clean_request_id(bad)
        assert cleaned != bad and len(cleaned) == 32


def test_request_id_round_trips_and_episode_events_are_logged(test_client, caplog):
    caplog.set_level(logging.INFO)
    r = test_client.post("/reset", json={"task_id": "phishing", "seed": 3}, headers={"X-Request-ID": "trace-1"})
    assert r.headers["X-Request-ID"] == "trace-1"
    assert test_client.get("/health").headers["X-Request-ID"]  # generated when absent

    test_client.post("/step", json={"action_type": "submit_investigation"}, headers={"X-Request-ID": "trace-2"})
    by_msg = {rec.getMessage(): rec for rec in caplog.records}
    reset = by_msg["episode.reset"]
    assert reset.task_id == "phishing" and reset.seed == 3
    done = by_msg["episode.complete"]
    assert done.task_id == "phishing" and done.steps == 1 and done.task_score is not None
    access = [rec for rec in caplog.records if rec.getMessage() == "http.request" and rec.path == "/reset"]
    assert access and access[0].status == 200
