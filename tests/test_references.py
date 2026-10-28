"""Checks on report/final/references.yaml (PHASE3_SPEC 6.3): one entry per citation key, each
IEEE string copied verbatim from the verified list in docs/BUILD_SPEC.md section 9.2."""

from __future__ import annotations

import yaml

from har.utils.paths import REPO_ROOT

REFERENCES = REPO_ROOT / "report" / "final" / "references.yaml"
BUILD_SPEC = REPO_ROOT / "docs" / "BUILD_SPEC.md"

# PHASE3_SPEC 6.3, in table order
KEYS = [
    "anguita2013", "reyesortiz2016", "mahmud2020", "dirgova2022", "yang2015", "jiang2015",
    "ronao2016", "wang2019", "xia2020", "ordonez2016", "hammerla2016", "murad2017", "zhao2018",
    "ucihar2013", "hochreiter1997", "schuster1997", "cho2014", "vaswani2017",
]


def _references() -> dict[str, str]:
    with REFERENCES.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_spec_9_2_lines() -> set[str]:
    """Every line of BUILD_SPEC 9.2, with a leading list marker removed."""
    lines = BUILD_SPEC.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("### 9.2"))
    end = next(i for i, line in enumerate(lines) if line.startswith("## 10."))
    return {line.removeprefix("- ") for line in lines[start:end]}


def test_one_entry_per_citation_key_in_spec_order() -> None:
    assert list(_references()) == KEYS


def test_every_string_is_copied_verbatim_from_build_spec() -> None:
    verified = _build_spec_9_2_lines()
    for key, value in _references().items():
        assert value in verified, f"{key} does not match its BUILD_SPEC 9.2 entry"


def test_no_two_keys_share_a_reference() -> None:
    values = list(_references().values())
    assert len(set(values)) == len(values)
