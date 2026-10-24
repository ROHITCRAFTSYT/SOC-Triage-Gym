"""Version single-sourcing.

__version__ used to be a hardcoded literal ("0.1.0") that had drifted from
pyproject ("0.3.0"). It's now derived from installed package metadata; the CLI
'version' command reports the same source. These stay lightweight (no heavy
top-level package import).
"""

import re
from pathlib import Path

import cli

ROOT = Path(__file__).resolve().parents[1]


def test_package_version_is_a_semver_string():
    v = cli._package_version()
    assert re.match(r"^\d+\.\d+\.\d+", v), v


def test_init_no_longer_hardcodes_a_drifting_version():
    src = (ROOT / "__init__.py").read_text(encoding="utf-8")
    assert '__version__ = "0.1.0"' not in src   # the old drifted literal is gone
    assert "_pkg_version" in src                 # derived from package metadata


def test_version_subcommand_parses_and_runs(capsys):
    ns = cli.build_parser().parse_args(["version"])
    assert ns.command == "version"
    assert ns.func is cli._cmd_version
    assert cli._cmd_version(ns) == 0
    assert "soc-gym" in capsys.readouterr().out
