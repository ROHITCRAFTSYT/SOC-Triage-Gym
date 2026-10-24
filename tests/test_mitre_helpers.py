"""Parent/sub-technique lookup helpers over the MITRE ATT&CK database.

These pin the technique-hierarchy navigation exposed from the ``data`` package:
resolving a sub-technique to its base technique and enumerating a base
technique's sub-techniques, including the punctuation-tolerant input path
shared with grading.
"""

from data import (
    TECHNIQUES,
    get_parent_technique,
    get_sub_techniques,
)


def test_parent_of_subtechnique_uses_explicit_key():
    assert get_parent_technique("T1566.001") == "T1566"
    assert get_parent_technique("T1078.002") == "T1078"


def test_parent_of_base_technique_is_none():
    assert get_parent_technique("T1566") is None
    assert get_parent_technique("T1190") is None


def test_parent_of_unknown_is_none():
    assert get_parent_technique("T9999.999") is None
    assert get_parent_technique("garbage") is None


def test_parent_tolerates_messy_input():
    # Shares normalization with grading, so wrapped/cased input still resolves.
    assert get_parent_technique("(t1566.001)") == "T1566"
    assert get_parent_technique("MITRE T1566.001.") == "T1566"


def test_sub_techniques_are_sorted_children():
    subs = get_sub_techniques("T1566")
    assert "T1566.001" in subs
    assert "T1566.002" in subs
    assert subs == sorted(subs)
    # Every returned child really points back to the parent.
    assert all(get_parent_technique(s) == "T1566" for s in subs)


def test_sub_techniques_of_leaf_or_unknown_is_empty():
    assert get_sub_techniques("T1566.001") == []
    assert get_sub_techniques("T9999") == []


def test_every_declared_parent_exists():
    # Guards against a sub-technique referencing a base technique we never added.
    for tid, data in TECHNIQUES.items():
        parent = data.get("parent")
        if parent is not None:
            assert parent in TECHNIQUES, f"{tid} references missing parent {parent}"
