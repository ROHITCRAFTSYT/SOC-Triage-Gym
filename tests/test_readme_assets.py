"""Keep the README chart metadata honest.

scripts/gen_readme_assets.py hardcodes a small task table to stay dependency-
light (no server import at plot time). This test makes sure that table cannot
silently drift away from the authoritative TASKS list in server/app.py — a
stale chart is worse than no chart.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _load_asset_module():
    path = os.path.join(ROOT, "scripts", "gen_readme_assets.py")
    spec = importlib.util.spec_from_file_location("gen_readme_assets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # importable without matplotlib installed
    return module


def test_asset_tasks_match_server_tasks():
    from server.app import TASKS as SERVER_TASKS

    assets = _load_asset_module()
    server_by_id = {t["id"]: t for t in SERVER_TASKS}
    asset_by_id = {row[0]: row for row in assets.TASKS}

    # Same set of task ids in both places.
    assert set(asset_by_id) == set(server_by_id), (
        "gen_readme_assets.TASKS is out of sync with server.app.TASKS"
    )

    # max_steps (index 2 in the asset tuple) must match the server metadata.
    for task_id, row in asset_by_id.items():
        assert row[2] == server_by_id[task_id]["max_steps"], (
            f"max_steps mismatch for {task_id}: "
            f"chart={row[2]} vs server={server_by_id[task_id]['max_steps']}"
        )


def test_asset_modes_are_known():
    assets = _load_asset_module()
    for row in assets.TASKS:
        mode = row[4]
        assert mode in assets.MODE_COLOR, f"unknown mode '{mode}' in {row[0]}"
