"""The `soc-gym techniques` CLI subcommand.

Drives the command through ``cli.main`` (not the heavier server-backed
subcommands) to confirm it lists the MITRE catalog, filters by tactic, emits
valid JSON, and rejects an unknown tactic with a non-zero exit code.
"""

import json

import cli


def test_lists_full_catalog(capsys):
    rc = cli.main(["techniques"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "T1059" in out
    assert "technique(s)." in out


def test_tactic_filter_narrows_output(capsys):
    rc = cli.main(["techniques", "--tactic", "execution"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "T1059" in out          # execution technique
    assert "T1566" not in out      # initial-access technique excluded


def test_json_output_is_parseable(capsys):
    rc = cli.main(["techniques", "--tactic", "execution", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert "T1059" in payload
    assert all(v["tactic"] == "execution" for v in payload.values())


def test_unknown_tactic_exits_nonzero(capsys):
    rc = cli.main(["techniques", "--tactic", "not-a-tactic"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Unknown tactic" in out
