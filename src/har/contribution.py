"""Contribution log generated from the git history of ``main`` (BUILD_SPEC 11.1).

Writes ``docs/contribution_log.md``: a per-member summary table (non-merge commits, lines
added and removed excluding generated and bulk files, merged PRs when ``gh`` is available,
first and last commit dates), changes to generated files, merge commits (who merged which
PR) and each member's commits with links. Members are matched against ``configs/team.yaml``
by author name, GitHub username or GitHub noreply email; ``git log`` applies ``.mailmap``,
so a member can join their own identities there. Uses only ``git`` and optionally ``gh``.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from har.utils.config import load_yaml
from har.utils.paths import CONFIGS, REPO_ROOT

# Generated or bulk files: their line changes are reported separately, not as code written.
GENERATED_PREFIXES = ("results/", "docs/figures/", "report/template/")
GENERATED_FILES = {"docs/contribution_log.md", "docs/dataset_stats.md", "docs/BUILD_SPEC.md", "CLAUDE.md"}
GENERATED_SUFFIXES = (".png",)

_RS, _US = "\x1e", "\x1f"   # record and field separators in the git log format
_FIELDS = "%H%x1f%aN%x1f%aE%x1f%ad%x1f%s"


@dataclass
class FileChange:
    """One file in a commit's ``--numstat``; ``added``/``removed`` are 0 for binary files."""

    path: str
    added: int
    removed: int


@dataclass
class Commit:
    """One commit from ``git log`` (author name after ``.mailmap``)."""

    sha: str
    author: str
    email: str
    date: str          # ISO 8601 author date
    subject: str
    files: list[FileChange] = field(default_factory=list)


@dataclass
class MemberStats:
    """Aggregates for one team member."""

    commits: list[Commit] = field(default_factory=list)
    added: int = 0
    removed: int = 0
    generated_changes: int = 0
    generated_added: int = 0
    generated_removed: int = 0
    top_level: Counter = field(default_factory=Counter)


def is_generated(path: str) -> bool:
    """True for generated or bulk files whose line counts are not credited as written code."""
    return (path in GENERATED_FILES or path.startswith(GENERATED_PREFIXES)
            or path.lower().endswith(GENERATED_SUFFIXES))


def run_git(args: list[str], cwd: Path = REPO_ROOT) -> str:
    """Run ``git <args>`` in ``cwd`` and return stdout (UTF-8). Raises ``RuntimeError`` on failure."""
    proc = subprocess.run(["git", "-c", "core.quotePath=false", *args], cwd=cwd,
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _header(line: str) -> Commit:
    sha, author, email, date, subject = line.split(_US, 4)
    return Commit(sha=sha.strip(), author=author.strip(), email=email.strip(), date=date.strip(),
                  subject=subject.strip().lstrip("﻿"))   # a stray BOM in a message is dropped


def parse_log(text: str) -> list[Commit]:
    """Parse ``git log --numstat`` output written with ``_RS`` before each commit header."""
    commits = []
    for record in text.split(_RS):
        lines = [line for line in record.splitlines() if line.strip()]
        if not lines:
            continue
        commit = _header(lines[0])
        for line in lines[1:]:
            added, removed, path = line.split("\t", 2)
            commit.files.append(FileChange(path, 0 if added == "-" else int(added),
                                           0 if removed == "-" else int(removed)))
        commits.append(commit)
    return commits


def read_commits(ref: str = "main", cwd: Path = REPO_ROOT) -> list[Commit]:
    """Non-merge commits reachable from ``ref``, oldest first, with per-file line counts."""
    text = run_git(["log", ref, "--no-merges", "--no-renames", "--numstat", "--reverse",
                    "--date=iso-strict", f"--pretty=format:{_RS}{_FIELDS}"], cwd)
    return parse_log(text)


def read_merges(ref: str = "main", cwd: Path = REPO_ROOT) -> list[Commit]:
    """Merge commits reachable from ``ref``, oldest first (no file list)."""
    text = run_git(["log", ref, "--merges", "--reverse", "--date=iso-strict",
                    f"--pretty=format:{_RS}{_FIELDS}"], cwd)
    return parse_log(text)


def load_members(team_file: Path = CONFIGS / "team.yaml") -> list[dict]:
    """Members from ``team.yaml``: dicts with ``id``, ``name`` and ``github``."""
    return list(load_yaml(team_file)["members"])


def match_member(author: str, email: str, members: list[dict]) -> str | None:
    """Member id for an author, by full name, GitHub username, or GitHub noreply email."""
    author_key = author.strip().lower()
    login = re.match(r"^(?:\d+\+)?([^@]+)@users\.noreply\.github\.com$", email.strip().lower())
    for member in members:
        github = str(member.get("github") or "").lower()
        if author_key == member["name"].lower() or (github and author_key == github):
            return member["id"]
        if github and login and login.group(1) == github:
            return member["id"]
    return None


def count_merged_prs(prs_json: str, members: list[dict]) -> dict[str, int]:
    """PRs per member id from ``gh pr list --json author`` output (matched by GitHub username)."""
    by_login = {str(m.get("github") or "").lower(): m["id"] for m in members if m.get("github")}
    counts = {m["id"]: 0 for m in members}
    for pr in json.loads(prs_json):
        member = by_login.get(str((pr.get("author") or {}).get("login", "")).lower())
        if member:
            counts[member] += 1
    return counts


def merged_prs(members: list[dict], base: str = "main", cwd: Path = REPO_ROOT) -> dict[str, int] | None:
    """Merged PRs into ``base`` per member via ``gh``; ``None`` if ``gh`` is unavailable or fails."""
    try:
        proc = subprocess.run(["gh", "pr", "list", "--state", "merged", "--base", base, "--limit", "1000",
                               "--json", "author,number,title"], cwd=cwd, capture_output=True,
                              text=True, encoding="utf-8", timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return count_merged_prs(proc.stdout, members)


def summarise(commits: list[Commit], members: list[dict]) -> tuple[dict[str, MemberStats], list[Commit]]:
    """Per-member statistics, plus the commits that match no member."""
    stats = {m["id"]: MemberStats() for m in members}
    unmatched = []
    for commit in commits:
        member = match_member(commit.author, commit.email, members)
        if member is None:
            unmatched.append(commit)
            continue
        s = stats[member]
        s.commits.append(commit)
        for change in commit.files:
            s.top_level[change.path.split("/")[0] + ("/" if "/" in change.path else "")] += 1
            if is_generated(change.path):
                s.generated_changes += 1
                s.generated_added += change.added
                s.generated_removed += change.removed
            else:
                s.added += change.added
                s.removed += change.removed
    return stats, unmatched


def https_repo_url(remote: str) -> str:
    """``git@github.com:o/r.git`` or ``https://github.com/o/r.git`` -> ``https://github.com/o/r``."""
    url = remote.strip()
    ssh = re.match(r"^git@([^:]+):(.+)$", url)
    if ssh:
        url = f"https://{ssh.group(1)}/{ssh.group(2)}"
    return re.sub(r"\.git$", "", url).rstrip("/")


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _sha_link(sha: str, repo_url: str | None) -> str:
    return f"[{sha[:7]}]({repo_url}/commit/{sha})" if repo_url else f"`{sha[:7]}`"


def render_markdown(stats: dict[str, MemberStats], unmatched: list[Commit], merges: list[Commit],
                    members: list[dict], head: str, ref: str, repo_url: str | None,
                    prs: dict[str, int] | None, generated_at: datetime) -> str:
    """The ``contribution_log.md`` text. The first table is the per-member summary."""
    label = {m["id"]: f"{m['name']} ({m['id']})" for m in members}
    lines = [
        "# Contribution log",
        "",
        f"Generated by `python scripts/contribution_log.py` on {generated_at:%Y-%m-%d %H:%M %z} "
        f"from the `{ref}` branch. Do not edit by hand.",
        "",
        f"- HEAD: `{head}`",
        f"- Repository: {repo_url or 'unknown'}",
        "",
        "## Summary per member",
        "",
    ]
    header = ["Member", "Commits", "Lines added", "Lines removed"] + (["PRs merged"] if prs is not None else [])
    lines += ["| " + " | ".join(header + ["First commit", "Last commit"]) + " |",
              "|---|" + "---:|" * (len(header) - 1) + "---|---|"]
    for member in members:
        s = stats[member["id"]]
        dates = [c.date[:10] for c in s.commits]
        row = [label[member["id"]], str(len(s.commits)), f"{s.added:,}", f"{s.removed:,}"]
        if prs is not None:
            row.append(str(prs.get(member["id"], 0)))
        row += [dates[0] if dates else "", dates[-1] if dates else ""]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "Commits exclude merge commits. Line counts exclude generated and bulk files (`results/`, "
        "`docs/figures/`, `docs/contribution_log.md`, `docs/dataset_stats.md`, `docs/BUILD_SPEC.md`, "
        "`CLAUDE.md`, `report/template/`, `*.png`); changes to those are counted in the next table."
        + (" PRs merged counts pull requests authored by the member and merged into `main` (from `gh`)."
           if prs is not None else " PR counts are omitted because `gh` was not available."),
        "",
        "## Generated and documentation files changed",
        "",
        "| Member | File changes | Lines added | Lines removed |",
        "|---|---:|---:|---:|",
    ]
    for member in members:
        s = stats[member["id"]]
        lines.append(f"| {label[member['id']]} | {s.generated_changes} | {s.generated_added:,} | "
                     f"{s.generated_removed:,} |")

    lines += ["", "## Merge commits", "", "| Date | Commit | Merged by | Subject |", "|---|---|---|---|"]
    for merge in merges:
        who = match_member(merge.author, merge.email, members)
        lines.append(f"| {merge.date[:10]} | {_sha_link(merge.sha, repo_url)} | "
                     f"{label[who] if who else _cell(merge.author)} | {_cell(merge.subject)} |")

    for member in members:
        s = stats[member["id"]]
        lines += ["", f"## {label[member['id']]}", ""]
        if not s.commits:
            lines.append("No commits yet.")
            continue
        paths = ", ".join(f"`{p}` ({n})" for p, n in sorted(s.top_level.items(), key=lambda kv: (-kv[1], kv[0])))
        lines += [f"Top-level paths changed (file changes): {paths}", "",
                  "| Date | Commit | Subject |", "|---|---|---|"]
        lines += [f"| {c.date[:10]} | {_sha_link(c.sha, repo_url)} | {_cell(c.subject)} |" for c in s.commits]

    if unmatched:
        lines += ["", "## Commits by authors not in configs/team.yaml", "",
                  "| Date | Commit | Author | Subject |", "|---|---|---|---|"]
        lines += [f"| {c.date[:10]} | {_sha_link(c.sha, repo_url)} | {_cell(c.author)} | {_cell(c.subject)} |"
                  for c in unmatched]
    return "\n".join(lines) + "\n"


def write_contribution_log(out: Path, ref: str = "main", repo_root: Path = REPO_ROOT,
                           team_file: Path = CONFIGS / "team.yaml", use_gh: bool = True) -> Path:
    """Generate the contribution log for ``ref`` in ``repo_root`` and write it to ``out``."""
    members = load_members(team_file)
    head = run_git(["rev-parse", ref], repo_root).strip()
    try:
        repo_url = https_repo_url(run_git(["remote", "get-url", "origin"], repo_root))
    except RuntimeError:
        repo_url = None
    stats, unmatched = summarise(read_commits(ref, repo_root), members)
    prs = merged_prs(members, base=ref, cwd=repo_root) if use_gh else None
    text = render_markdown(stats, unmatched, read_merges(ref, repo_root), members, head, ref,
                           repo_url, prs, datetime.now().astimezone())
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    return out
