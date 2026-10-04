"""Tests for har.contribution and scripts/contribution_log.py.

The end-to-end tests build a throwaway git repository with fictional authors (never the
project repository), so they need ``git`` but no network and no ``gh``.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from har.contribution import (
    Commit, FileChange, count_merged_prs, https_repo_url, is_generated, match_member, parse_log,
    render_markdown, summarise, write_contribution_log,
)
from har.utils.paths import REPO_ROOT

MEMBERS = [
    {"id": "M1", "name": "Alice Example", "github": "alice-gh"},
    {"id": "M2", "name": "Bob Example", "github": "bob-gh"},
]
RS, US = "\x1e", "\x1f"


# ---------------------------------------------------------------- units

@pytest.mark.parametrize("path, generated", [
    ("results/mlp/run/metrics.json", True), ("docs/figures/a.png", True), ("docs/BUILD_SPEC.md", True),
    ("CLAUDE.md", True), ("docs/contribution_log.md", True), ("docs/dataset_stats.md", True),
    ("report/template/t.docx", True), ("notebooks/plot.PNG", True),
    ("src/har/data/raw.py", False), ("docs/lit/addyan.md", False), ("results.py", False),
])
def test_is_generated(path: str, generated: bool) -> None:
    assert is_generated(path) is generated


def test_parse_log_reads_numstat_binary_files_and_strips_a_bom() -> None:
    text = (
        f"{RS}aaa111{US}Alice Example{US}a@x{US}2026-10-03T10:00:00+05:30{US}﻿feat: one\n\n"
        "10\t2\tsrc/a.py\n-\t-\tdocs/figures/f.png\n"
        f"{RS}bbb222{US}Bob Example{US}b@x{US}2026-10-04T11:00:00+05:30{US}fix: a | b\n"
    )
    commits = parse_log(text)
    assert [c.sha for c in commits] == ["aaa111", "bbb222"]
    assert commits[0].subject == "feat: one"
    assert commits[0].files == [FileChange("src/a.py", 10, 2), FileChange("docs/figures/f.png", 0, 0)]
    assert commits[1].files == [] and commits[1].subject == "fix: a | b"


def test_match_member_by_name_login_or_noreply_email() -> None:
    assert match_member("Alice Example", "alice@personal.example", MEMBERS) == "M1"
    assert match_member("alice example", "x@y", MEMBERS) == "M1"
    assert match_member("bob-gh", "bob@personal.example", MEMBERS) == "M2"   # GitHub merge button
    assert match_member("B.", "12345+bob-gh@users.noreply.github.com", MEMBERS) == "M2"
    assert match_member("Carol Outside", "carol@x", MEMBERS) is None


def test_count_merged_prs_by_github_login() -> None:
    prs = json.dumps([{"author": {"login": "Alice-GH"}, "number": 1},
                      {"author": {"login": "bob-gh"}, "number": 2},
                      {"author": {"login": "bob-gh"}, "number": 3},
                      {"author": {"login": "someone"}, "number": 4}])
    assert count_merged_prs(prs, MEMBERS) == {"M1": 1, "M2": 2}


@pytest.mark.parametrize("remote", ["git@github.com:o/r.git", "https://github.com/o/r.git",
                                    "https://github.com/o/r/\n"])
def test_https_repo_url(remote: str) -> None:
    assert https_repo_url(remote) == "https://github.com/o/r"


def _commit(sha: str, author: str, files: list[FileChange], date: str = "2026-10-04T09:00:00+00:00") -> Commit:
    return Commit(sha=sha, author=author, email="", date=date, subject=f"subject {sha}", files=files)


def test_summarise_separates_generated_lines_and_unmatched_authors() -> None:
    commits = [
        _commit("a1", "Alice Example", [FileChange("src/x.py", 5, 1), FileChange("results/r.json", 100, 0)]),
        _commit("a2", "Alice Example", [FileChange("README.md", 2, 2)]),
        _commit("c1", "Carol Outside", [FileChange("src/y.py", 9, 0)]),
    ]
    stats, unmatched = summarise(commits, MEMBERS)
    alice = stats["M1"]
    assert len(alice.commits) == 2 and (alice.added, alice.removed) == (7, 3)
    assert (alice.generated_changes, alice.generated_added) == (1, 100)
    assert alice.top_level == {"src/": 1, "results/": 1, "README.md": 1}
    assert stats["M2"].commits == []
    assert [c.sha for c in unmatched] == ["c1"]


def test_render_puts_the_summary_table_first_and_records_head() -> None:
    commits = [_commit("a" * 40, "Alice Example", [FileChange("src/x.py", 3, 0)])]
    stats, unmatched = summarise(commits, MEMBERS)
    text = render_markdown(stats, unmatched, [], MEMBERS, "f" * 40, "main", "https://github.com/o/r",
                           {"M1": 4, "M2": 0}, datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc))
    first_table = next(line for line in text.splitlines() if line.startswith("|"))
    assert first_table.startswith("| Member | Commits | Lines added | Lines removed | PRs merged |")
    assert "| Alice Example (M1) | 1 | 3 | 0 | 4 | 2026-10-04 | 2026-10-04 |" in text
    assert "| Bob Example (M2) | 0 | 0 | 0 | 0 |  |  |" in text
    # The report builder finds the sha with this pattern.
    assert re.search(r"HEAD\W+([0-9a-f]{7,40})", text).group(1) == "f" * 40
    assert f"[aaaaaaa](https://github.com/o/r/commit/{'a' * 40})" in text
    assert "—" not in text


# ---------------------------------------------------------------- end to end on a temp repository

def _git(repo: Path, *args: str, author: tuple[str, str] | None = None) -> str:
    identity = ["-c", f"user.name={author[0]}", "-c", f"user.email={author[1]}"] if author else []
    return subprocess.run(["git", *identity, *args], cwd=repo, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def _commit_file(repo: Path, rel: str, content: str, message: str, author: tuple[str, str]) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _git(repo, "add", rel)
    _git(repo, "commit", "-q", "-m", message, author=author)


@pytest.fixture()
def repo(tmp_path: Path) -> tuple[Path, Path]:
    """Temp repo: Alice (two identities joined by .mailmap), Bob on a merged branch, Carol unknown."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "remote", "add", "origin", "git@github.com:o/r.git")
    alice, alice_old, bob = ("Alice Example", "alice@x"), ("A. Ex", "old-alice@x"), ("Bob Example", "bob@x")
    _commit_file(repo, ".mailmap", "Alice Example <alice@x> A. Ex <old-alice@x>\n", "chore: mailmap", alice)
    _commit_file(repo, "src/a.py", "a\nb\nc\n", "feat: a", alice_old)
    _commit_file(repo, "results/m/run/metrics.json", "{}\n" * 50, "exp: run", alice)
    _git(repo, "switch", "-q", "-c", "bob/feature")
    _commit_file(repo, "tests/test_b.py", "x\ny\n", "test: b", bob)
    _git(repo, "switch", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "bob/feature", "-m", "Merge pull request #1 from o/bob/feature",
         author=bob)
    _commit_file(repo, "notes.txt", "n\n", "docs: notes", ("Carol Outside", "carol@x"))
    team = tmp_path / "team.yaml"
    team.write_text(yaml.safe_dump({"members": MEMBERS}), encoding="utf-8")
    return repo, team


def test_end_to_end_log_of_a_temp_repository(repo: tuple[Path, Path], tmp_path: Path) -> None:
    repo_dir, team = repo
    out = write_contribution_log(tmp_path / "out" / "log.md", ref="main", repo_root=repo_dir,
                                 team_file=team, use_gh=False)
    text = out.read_text(encoding="utf-8")
    head = _git(repo_dir, "rev-parse", "main").strip()
    assert f"HEAD: `{head}`" in text
    # .mailmap joins Alice's two identities; the results/ file is not counted as written lines.
    assert "| Alice Example (M1) | 3 | 4 | 0 |" in text
    assert "| Bob Example (M2) | 1 | 2 | 0 |" in text
    assert "PRs merged" not in text and "`gh` was not available" in text
    assert "| Alice Example (M1) | 1 | 50 | 0 |" in text          # generated-files table
    assert "Bob Example (M2) | Merge pull request #1 from o/bob/feature |" in text
    assert "https://github.com/o/r/commit/" in text
    assert "Carol Outside | docs: notes |" in text
    # Alice's commits are listed oldest first.
    alice_section = text.split("## Alice Example (M1)")[1].split("## ")[0]
    assert alice_section.index("chore: mailmap") < alice_section.index("feat: a") < alice_section.index("exp: run")


def test_script_main_passes_its_arguments(repo: tuple[Path, Path], tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("contribution_log", REPO_ROOT / "scripts" / "contribution_log.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo_dir, team = repo
    out = module.main(["--repo", str(repo_dir), "--team-file", str(team), "--out", str(tmp_path / "c.md"),
                       "--no-gh"])
    assert out == tmp_path / "c.md" and out.is_file()
