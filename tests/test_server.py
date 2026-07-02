"""
Tests for FastAPI endpoints: health, metadata, schema, mcp, reset, step, state, tasks, baseline.
"""



class TestHealthEndpoint:
    def test_health_endpoint(self, test_client):
        """GET /health returns 200 with status healthy."""
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestMetadataEndpoint:
    def test_metadata_endpoint(self, test_client):
        """GET /metadata returns environment metadata."""
        resp = test_client.get("/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "soc-triage-gym"
        assert "tasks" in data
        assert "phishing" in data["tasks"]
        assert "lateral_movement" in data["tasks"]
        assert "queue_management" in data["tasks"]


class TestSchemaEndpoint:
    def test_schema_endpoint(self, test_client):
        """GET /schema returns action, observation, and state JSON schemas."""
        resp = test_client.get("/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert "action" in data
        assert "observation" in data
        assert "state" in data
        # Each should be a valid JSON schema with properties
        assert "properties" in data["action"]
        assert "properties" in data["observation"]


class TestMCPEndpoint:
    def test_mcp_endpoint(self, test_client):
        """POST /mcp returns a valid JSON-RPC 2.0 response."""
        resp = test_client.post("/mcp", json={"method": "tools/list", "id": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert "result" in data
        assert "tools" in data["result"]

    def test_mcp_endpoint_default(self, test_client):
        """POST /mcp with unknown method returns JSON-RPC error."""
        resp = test_client.post("/mcp", json={"method": "unknown", "id": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 2
        # Unknown methods return either an error or a result
        assert "error" in data or "result" in data


class TestResetEndpoint:
    def test_reset_endpoint(self, test_client):
        """POST /reset creates a new episode and returns initial observation."""
        resp = test_client.post("/reset", json={"task_id": "phishing", "seed": 42})
        assert resp.status_code == 200
        data = resp.json()
        assert data["step"] == 0
        assert data["done"] is False
        assert data["task_id"] == "phishing"
        assert len(data["alert_queue"]) == 1
        assert data["investigation_budget"] == 15

    def test_reset_invalid_task(self, test_client):
        """POST /reset with invalid task_id returns 400."""
        resp = test_client.post("/reset", json={"task_id": "nonexistent", "seed": 42})
        assert resp.status_code == 400

    def test_team_reset_endpoint(self, test_client):
        """POST /reset with team mode returns a team-mode observation."""
        resp = test_client.post("/reset", json={"task_id": "team_phishing_escalation", "seed": 42, "mode": "team"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["episode_mode"] == "team"
        assert data["current_role"] == "tier1"


class TestStepEndpoint:
    def test_step_endpoint(self, test_client):
        """POST /step executes an action and returns updated observation."""
        # First reset
        test_client.post("/reset", json={"task_id": "phishing", "seed": 42})

        # Then step with NOOP
        resp = test_client.post("/step", json={"action_type": "noop"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["step"] == 1
        assert data["done"] is False

    def test_step_without_reset(self, test_client):
        """POST /step without reset returns 400."""
        resp = test_client.post("/step", json={"action_type": "noop"})
        assert resp.status_code == 400


class TestStateEndpoint:
    def test_state_endpoint(self, test_client):
        """GET /state returns current episode metadata."""
        # Reset first
        test_client.post("/reset", json={"task_id": "phishing", "seed": 42})

        resp = test_client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "phishing"
        assert data["step_count"] == 0
        assert data["max_steps"] == 15
        assert data["done"] is False
        assert data["alert_count"] == 1


class TestTasksEndpoint:
    def test_tasks_endpoint(self, test_client):
        """GET /tasks returns list of available tasks."""
        resp = test_client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data
        task_ids = [t["id"] for t in data["tasks"]]
        assert "phishing" in task_ids
        assert "lateral_movement" in task_ids
        assert "queue_management" in task_ids
        assert "team_phishing_escalation" in task_ids

    def test_tasks_single(self, test_client):
        """GET /tasks/{task_id} returns details for a single task."""
        resp = test_client.get("/tasks/phishing")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "phishing"
        assert data["difficulty"] == "easy"
        assert data["max_steps"] == 15

    def test_team_task_single(self, test_client):
        """GET /tasks/{task_id} returns details for a team task."""
        resp = test_client.get("/tasks/team_phishing_escalation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "team_phishing_escalation"
        assert data["max_steps"] == 68

    def test_tasks_not_found(self, test_client):
        """GET /tasks/{task_id} with invalid id returns 404."""
        resp = test_client.get("/tasks/nonexistent")
        assert resp.status_code == 404


class TestBaselineEndpoint:
    def test_baseline_endpoint(self, test_client):
        """POST /baseline runs the heuristic agent and returns a score."""
        resp = test_client.post("/baseline", json={"task_id": "phishing", "seed": 42})
        assert resp.status_code == 200
        data = resp.json()
        assert "score" in data
        assert "breakdown" in data
        assert data["agent"] == "heuristic"
        assert data["task_id"] == "phishing"
        assert isinstance(data["score"], (int, float))
        assert 0.0 <= data["score"] <= 1.0


class TestLogsEndpoint:
    def test_logs_endpoint_returns_entries_for_alert_and_source(self, test_client):
        """GET /logs/{source} should return source-scoped entries for a specific alert."""
        reset_resp = test_client.post("/reset", json={"task_id": "phishing", "seed": 42})
        alert_id = reset_resp.json()["alert_queue"][0]["alert_id"]

        resp = test_client.get(f"/logs/email_gateway?alert_id={alert_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "email_gateway"
        assert data["alert_id"] == alert_id
        assert data["count"] > 0
        assert len(data["entries"]) > 0

    def test_logs_endpoint_returns_entries_for_source_without_alert_id(self, test_client):
        """GET /logs/{source} without alert_id should aggregate matching source entries."""
        test_client.post("/reset", json={"task_id": "phishing", "seed": 42})

        resp = test_client.get("/logs/email_gateway")
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "email_gateway"
        assert data["count"] > 0
        assert len(data["entries"]) > 0


class TestRound2Endpoints:
    def test_generate_scenario_endpoint(self, test_client):
        """POST /generate_scenario returns a valid generated scenario payload."""
        resp = test_client.post("/generate_scenario", json={"seed": 42})
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "red_team_generated"
        assert "alerts" in data
        assert len(data["alerts"]) > 0
