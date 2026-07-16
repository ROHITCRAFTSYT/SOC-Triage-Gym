"""
Tests for the production-hardening features added in v0.2.0:
multi-session concurrency, API-key auth, rate limiting, Prometheus
metrics, the episode audit trail, and the soc-gym CLI.
"""

import time

from server.sessions import DEFAULT_SESSION_ID, SessionManager

# ---------------------------------------------------------------------------
# Multi-session concurrency
# ---------------------------------------------------------------------------


class TestSessions:
    def test_sessions_are_isolated(self, test_client):
        """Two sessions run different tasks without clobbering each other."""
        r1 = test_client.post("/reset", json={"task_id": "phishing", "seed": 1}, headers={"X-Session-ID": "alice"})
        r2 = test_client.post(
            "/reset", json={"task_id": "queue_management", "seed": 2}, headers={"X-Session-ID": "bob"}
        )
        assert r1.status_code == 200 and r2.status_code == 200
        assert len(r1.json()["alert_queue"]) == 1
        assert len(r2.json()["alert_queue"]) == 20

        s1 = test_client.get("/state", headers={"X-Session-ID": "alice"}).json()
        s2 = test_client.get("/state", headers={"X-Session-ID": "bob"}).json()
        assert s1["task_id"] == "phishing"
        assert s2["task_id"] == "queue_management"

    def test_step_only_advances_own_session(self, test_client):
        test_client.post("/reset", json={"task_id": "phishing", "seed": 1}, headers={"X-Session-ID": "alice"})
        test_client.post("/reset", json={"task_id": "phishing", "seed": 1}, headers={"X-Session-ID": "bob"})
        test_client.post(
            "/step",
            json={"action_type": "query_logs", "log_source": "email_gateway"},
            headers={"X-Session-ID": "alice"},
        )
        s1 = test_client.get("/state", headers={"X-Session-ID": "alice"}).json()
        s2 = test_client.get("/state", headers={"X-Session-ID": "bob"}).json()
        assert s1["step_count"] == 1
        assert s2["step_count"] == 0

    def test_default_session_without_header(self, test_client):
        """No header keeps the original single-tenant behaviour."""
        resp = test_client.post("/reset", json={"task_id": "phishing", "seed": 42})
        assert resp.status_code == 200
        assert test_client.get("/state").json()["task_id"] == "phishing"

    def test_body_session_id_overrides_header(self, test_client):
        test_client.post("/reset", json={"task_id": "phishing", "seed": 1, "session_id": "carol"})
        sessions = {s["session_id"] for s in test_client.get("/sessions").json()["sessions"]}
        assert "carol" in sessions

    def test_invalid_session_id_rejected(self, test_client):
        resp = test_client.post("/reset", json={}, headers={"X-Session-ID": "bad id !!"})
        assert resp.status_code == 400

    def test_sessions_list_and_delete(self, test_client):
        test_client.post("/reset", json={}, headers={"X-Session-ID": "temp"})
        assert "temp" in {s["session_id"] for s in test_client.get("/sessions").json()["sessions"]}
        assert test_client.delete("/sessions/temp").status_code == 200
        assert test_client.delete("/sessions/temp").status_code == 404

    def test_ttl_eviction(self):
        mgr = SessionManager(max_sessions=10, ttl_seconds=1)
        mgr.get_or_create("old")
        mgr.peek("old").last_used = time.time() - 5
        mgr.get_or_create("new")
        assert mgr.peek("old") is None
        assert mgr.peek("new") is not None

    def test_lru_eviction_never_evicts_default(self):
        mgr = SessionManager(max_sessions=2, ttl_seconds=0)
        mgr.get_or_create(DEFAULT_SESSION_ID)
        mgr.get_or_create("a")
        mgr.peek("a").last_used = time.time() - 100
        mgr.get_or_create("b")  # over cap → evicts LRU non-default ("a")
        assert mgr.peek("a") is None
        assert mgr.peek(DEFAULT_SESSION_ID) is not None
        assert mgr.peek("b") is not None

    def test_validate_id(self):
        assert SessionManager.validate_id("trainer-1.a_B")
        assert not SessionManager.validate_id("")
        assert not SessionManager.validate_id("x" * 65)
        assert not SessionManager.validate_id("has space")


# ---------------------------------------------------------------------------
# API-key auth + rate limiting
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_auth_disabled_by_default(self, test_client):
        assert test_client.post("/reset", json={}).status_code == 200

    def test_auth_enforced_when_key_set(self, test_client, monkeypatch):
        monkeypatch.setenv("SOC_GYM_API_KEY", "sekrit")
        assert test_client.post("/reset", json={}).status_code == 401
        assert test_client.post("/reset", json={}, headers={"X-API-Key": "wrong"}).status_code == 401
        assert test_client.post("/reset", json={}, headers={"X-API-Key": "sekrit"}).status_code == 200
        assert test_client.post("/reset", json={}, headers={"Authorization": "Bearer sekrit"}).status_code == 200

    def test_health_and_landing_exempt_from_auth(self, test_client, monkeypatch):
        monkeypatch.setenv("SOC_GYM_API_KEY", "sekrit")
        assert test_client.get("/health").status_code == 200
        assert test_client.get("/").status_code == 200
        assert test_client.get("/docs").status_code == 200

    def test_rate_limit(self, test_client, monkeypatch):
        monkeypatch.setenv("SOC_GYM_RATE_LIMIT", "2")
        codes = [test_client.get("/state").status_code for _ in range(4)]
        assert codes[0] == 200 and codes[1] == 200
        assert 429 in codes[2:]
        retry = [c for c in codes if c == 429]
        assert retry  # limit enforced

    def test_rate_limit_disabled_by_default(self, test_client):
        codes = [test_client.get("/state").status_code for _ in range(30)]
        assert all(c == 200 for c in codes)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_metrics_endpoint_prometheus_format(self, test_client):
        test_client.post("/reset", json={"task_id": "phishing", "seed": 42})
        test_client.post("/step", json={"action_type": "submit_investigation"})
        text = test_client.get("/metrics").text
        assert "# TYPE socgym_requests_total counter" in text
        assert 'socgym_episodes_started_total{task="phishing"} 1' in text
        assert 'socgym_steps_total{task="phishing"} 1' in text
        assert 'socgym_episodes_completed_total{task="phishing"} 1' in text
        assert "socgym_active_sessions" in text

    def test_metrics_use_route_template_not_raw_path(self, test_client):
        test_client.post("/reset", json={"task_id": "phishing", "seed": 42})
        test_client.get("/api/alerts/ALERT-DOES-NOT-EXIST")
        text = test_client.get("/metrics").text
        assert "/api/alerts/{alert_id}" in text
        assert "ALERT-DOES-NOT-EXIST" not in text


# ---------------------------------------------------------------------------
# Episode audit trail
# ---------------------------------------------------------------------------


class TestAuditTrail:
    def test_episode_recorded_and_replayable(self, test_client):
        reset = test_client.post("/reset", json={"task_id": "phishing", "seed": 42}).json()
        test_client.post("/step", json={"action_type": "query_logs", "log_source": "email_gateway"})
        test_client.post("/step", json={"action_type": "submit_investigation"})

        episodes = test_client.get("/episodes").json()["episodes"]
        assert len(episodes) == 1
        ep = episodes[0]
        assert ep["task_id"] == "phishing"
        assert ep["done"] is True
        assert ep["event_count"] == 3  # reset + 2 steps

        trace = test_client.get(f"/episodes/{ep['episode_id']}/trace").json()
        assert trace["events"][0]["type"] == "reset"
        assert trace["events"][1]["action"]["action_type"] == "query_logs"
        assert trace["events"][-1]["done"] is True
        assert isinstance(trace["events"][-1]["cumulative_reward"], float)
        # episode_id in the trace matches the observation stream
        assert ep["episode_id"] == reset.get("episode_id", ep["episode_id"])

    def test_jsonl_export(self, test_client):
        test_client.post("/reset", json={"task_id": "phishing", "seed": 42})
        ep = test_client.get("/episodes").json()["episodes"][0]
        resp = test_client.get(f"/episodes/{ep['episode_id']}/trace?format=jsonl")
        assert resp.status_code == 200
        lines = resp.text.strip().splitlines()
        assert len(lines) == 2  # header + reset event
        import json

        assert json.loads(lines[0])["type"] == "episode"

    def test_unknown_episode_404(self, test_client):
        assert test_client.get("/episodes/nope/trace").status_code == 404

    def test_episodes_filter_by_session(self, test_client):
        test_client.post("/reset", json={}, headers={"X-Session-ID": "alice"})
        test_client.post("/reset", json={}, headers={"X-Session-ID": "bob"})
        eps = test_client.get("/episodes?session_id=alice").json()["episodes"]
        assert len(eps) == 1
        assert eps[0]["session_id"] == "alice"

    def test_audit_window_bounded(self):
        from server.audit import AuditTrail

        trail = AuditTrail(max_episodes=3)
        for i in range(5):
            trail.start_episode(f"ep{i}", "s", "phishing", 42, "tier1_solo")
        assert len(trail.list_episodes(limit=100)) == 3
        assert trail.get("ep0") is None
        assert trail.get("ep4") is not None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_parser_subcommands(self):
        from cli import build_parser

        parser = build_parser()
        for argv in (["tasks"], ["serve"], ["demo"], ["benchmark"], ["validate"]):
            args = parser.parse_args(argv)
            assert callable(args.func)

    def test_tasks_command_output(self, capsys):
        from cli import main

        assert main(["tasks"]) == 0
        out = capsys.readouterr().out
        assert "phishing" in out
        assert "apt_campaign" in out

    def test_tasks_json_output(self, capsys):
        import json

        from cli import main

        assert main(["tasks", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert any(t["id"] == "red_team_generated" for t in data)


# ---------------------------------------------------------------------------
# SDK client construction
# ---------------------------------------------------------------------------


class TestClientSDK:
    def test_headers_wired(self):
        from client import SOCTriageClient

        c = SOCTriageClient("http://x", session_id="t1", api_key="k")
        try:
            assert c._client.headers["X-Session-ID"] == "t1"
            assert c._client.headers["Authorization"] == "Bearer k"
        finally:
            c.close()

    def test_defaults_backward_compatible(self):
        from client import SOCTriageClient

        c = SOCTriageClient()
        try:
            assert c.base_url == "http://localhost:8000"
            assert "X-Session-ID" not in c._client.headers
        finally:
            c.close()
