"""MITRE technique-ID normalization.

Grading mapped techniques by exact string equality silently dropped correct
answers that differ only in case or an emitted prefix. normalize_technique_id
canonicalizes them; these pin the normalization and confirm is_valid_technique
now accepts the canonical forms.
"""

from data.mitre_attack import is_valid_technique, normalize_technique_id


def test_uppercase_and_strip():
    assert normalize_technique_id("  t1566.001  ") == "T1566.001"
    assert normalize_technique_id("T1078") == "T1078"


def test_strips_common_prefixes():
    assert normalize_technique_id("MITRE T1566") == "T1566"
    assert normalize_technique_id("ATT&CK T1078.002") == "T1078.002"
    assert normalize_technique_id("technique: t1190") == "T1190"


def test_unknown_passes_through_uppercased():
    assert normalize_technique_id("not-a-technique") == "NOT-A-TECHNIQUE"


def test_empty_and_none_safe():
    assert normalize_technique_id("") == ""
    assert normalize_technique_id(None) == ""  # type: ignore[arg-type]


def test_is_valid_technique_is_now_case_insensitive():
    assert is_valid_technique("t1566.001")
    assert is_valid_technique("  T1566.001 ")
    assert is_valid_technique("MITRE T1078")
    assert not is_valid_technique("T9999.999")
