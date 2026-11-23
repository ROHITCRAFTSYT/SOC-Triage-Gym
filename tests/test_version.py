"""Version single-sourcing.

__version__ and 'soc-gym version' read the installed package metadata, and
fall back to a literal only when running from a source checkout. These tests
pin that behaviour without importing the heavy top-level package.
"""

import ast
import importlib.metadata
import re
from pathlib import Path

import pytest

import cli

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert match, "pyproject.toml has no version"
    return match.group(1)


def _init_fallback_version() -> str:
    tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "__version__" for t in stmt.targets)
                    and isinstance(stmt.value, ast.Constant)
                ):
                    return stmt.value.value
    raise AssertionError("__init__.py has no fallback __version__ assignment")


def test_version_comes_from_package_metadata(monkeypatch):
    seen = []

    def fake_version(name):
        seen.append(name)
        return "9.8.7"

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    assert cli._package_version() == "9.8.7"
    assert seen == ["soc-triage-gym"]


def test_version_falls_back_to_pyproject_version_without_install(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert cli._package_version() == _pyproject_version()


def test_init_fallback_matches_pyproject():
    assert _init_fallback_version() == _pyproject_version()


@pytest.mark.parametrize("installed", ["9.8.7", "0.3.0"])
def test_version_subcommand_prints_the_package_version(monkeypatch, capsys, installed):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: installed)
    ns = cli.build_parser().parse_args(["version"])
    assert ns.func is cli._cmd_version
    assert ns.func(ns) == 0
    assert capsys.readouterr().out == f"soc-gym {installed}\n"
