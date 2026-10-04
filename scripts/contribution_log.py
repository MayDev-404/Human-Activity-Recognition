"""Generate docs/contribution_log.md from the git history of main.

Usage: python scripts/contribution_log.py [--ref main] [--no-gh]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from har.contribution import write_contribution_log
from har.utils.paths import CONFIGS, REPO_ROOT


def main(argv: list[str] | None = None) -> Path:
    """Parse arguments, write the contribution log and return its path."""
    parser = argparse.ArgumentParser(description="Contribution log from the git history.")
    parser.add_argument("--ref", default="main", help="branch or commit to describe (default: main)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "contribution_log.md")
    parser.add_argument("--team-file", type=Path, default=CONFIGS / "team.yaml")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT, help="repository to read")
    parser.add_argument("--no-gh", action="store_true", help="skip merged-PR counts from gh")
    args = parser.parse_args(argv)
    out = write_contribution_log(args.out, ref=args.ref, repo_root=args.repo,
                                 team_file=args.team_file, use_gh=not args.no_gh)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
