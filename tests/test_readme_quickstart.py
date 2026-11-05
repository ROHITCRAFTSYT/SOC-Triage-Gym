"""The README's Python quick-start must keep running against the real API."""

import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_readme_python_quickstart_runs(capsys):
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    match = re.search(r"Use the environment in-process \(Python API\):\*\*\n\n```python\n(.*?)```", readme, re.S)
    assert match, "README Python quick-start block not found"
    exec(compile(match.group(1), "README.md", "exec"), {})
    assert capsys.readouterr().out.strip()
