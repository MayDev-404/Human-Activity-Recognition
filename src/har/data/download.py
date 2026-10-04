"""Download, extract and verify the UCI HAR dataset (UCI id 240, DOI 10.24432/C54S4K).

CLI: ``python -m har.data.download [--force]``. Idempotent: a folder that already verifies is
left alone unless ``force`` is set. The current UCI archive holds a *nested* zip that contains
the ``UCI HAR Dataset`` folder; the legacy archive holds the folder directly. Both carry
``__MACOSX/`` and ``.DS_Store`` entries, which are skipped.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from tqdm import tqdm

from har.data.constants import ACTIVITY_NAMES, CHANNELS, N_FEATURES, WINDOW_LEN
from har.utils.paths import DATA_ROOT, UCI_DIR

# Tried in order: the current archive (about 61 MB, nested zip), then the legacy one.
URLS: tuple[str, ...] = (
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
)
DATASET_DIRNAME = "UCI HAR Dataset"
# Windows per official split (21 training subjects, 9 test subjects).
SPLIT_SIZES: dict[str, int] = {"train": 7352, "test": 2947}

MANUAL_HINT = (
    f"Download the zip from the URL above in a browser and place the extracted "
    f"'{DATASET_DIRNAME}' folder in data/raw/"
)
_CHUNK = 1 << 20


def expected_shapes() -> dict[str, tuple[int, int | None]]:
    """Map each file the pipelines read (path relative to the dataset folder) to (rows, columns).

    ``columns`` is ``None`` for the two name files, where only the row count is checked.
    """
    shapes: dict[str, tuple[int, int | None]] = {
        "activity_labels.txt": (len(ACTIVITY_NAMES), None),
        "features.txt": (N_FEATURES, None),
    }
    for split, n in SPLIT_SIZES.items():
        shapes[f"{split}/X_{split}.txt"] = (n, N_FEATURES)
        shapes[f"{split}/y_{split}.txt"] = (n, 1)
        shapes[f"{split}/subject_{split}.txt"] = (n, 1)
        for channel in CHANNELS:
            shapes[f"{split}/Inertial Signals/{channel}_{split}.txt"] = (n, WINDOW_LEN)
    return shapes


def text_matrix_shape(path: Path) -> tuple[int, int]:
    """(rows, columns) of a whitespace-separated text file, ignoring blank lines.

    Raises ``ValueError`` if the rows do not all have the same number of columns.
    """
    n_rows, n_cols = 0, 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            width = len(line.split())
            if width == 0:
                continue
            if n_rows == 0:
                n_cols = width
            elif width != n_cols:
                raise ValueError(f"{path}: line {line_no} has {width} columns, expected {n_cols}")
            n_rows += 1
    return n_rows, n_cols


def _count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def verify_dataset(uci_dir: Path = UCI_DIR) -> None:
    """Check that every expected file exists and has the expected shape.

    Raises ``FileNotFoundError`` listing any missing files, or ``ValueError`` listing every
    file whose row or column count is wrong.
    """
    uci_dir = Path(uci_dir)
    if not uci_dir.is_dir():
        raise FileNotFoundError(f"{uci_dir} does not exist; run python -m har.data.download")

    shapes = expected_shapes()
    missing = [rel for rel in shapes if not (uci_dir / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"{uci_dir} is missing {len(missing)} file(s): " + ", ".join(missing))

    problems = []
    for rel, (rows, cols) in shapes.items():
        path = uci_dir / rel
        if cols is None:
            got = (_count_rows(path), None)
        else:
            try:
                got = text_matrix_shape(path)
            except ValueError as exc:
                problems.append(str(exc))
                continue
        if got != (rows, cols):
            shown = f"{got[0]} rows" if cols is None else f"{got[0]} x {got[1]}"
            wanted = f"{rows} rows" if cols is None else f"{rows} x {cols}"
            problems.append(f"{rel}: {shown}, expected {wanted}")
    if problems:
        raise ValueError(f"{uci_dir} failed verification:\n  " + "\n  ".join(problems))


def _is_junk(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return "__MACOSX" in parts or (bool(parts) and parts[-1] == ".DS_Store")


def extract_archive(zip_path: Path, dest: Path, _depth: int = 0) -> Path:
    """Extract ``zip_path`` into ``dest``, skipping ``__MACOSX/`` and ``.DS_Store`` entries.

    Any zip found inside is extracted into ``dest`` as well and then deleted. Returns the path
    of the ``UCI HAR Dataset`` folder; raises ``FileNotFoundError`` if the archive has none.
    """
    dest = Path(dest)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not _is_junk(m.filename)]
        zf.extractall(dest, members=members)
    nested = [dest / m.filename for m in members if m.filename.lower().endswith(".zip") and not m.is_dir()]
    if _depth < 2:
        for inner in nested:
            extract_archive(inner, dest, _depth + 1)
            inner.unlink()
    if _depth > 0:
        return dest

    found = dest / DATASET_DIRNAME
    if not found.is_dir():
        candidates = [p for p in dest.rglob(DATASET_DIRNAME) if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No '{DATASET_DIRNAME}' folder inside {zip_path}")
        found = candidates[0]
    return found


def _fetch(url: str, target: Path) -> None:
    """Stream ``url`` into ``target`` with a progress bar."""
    request = urllib.request.Request(url, headers={"User-Agent": "har-dl-comparison/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc="UCI HAR") as bar:
            while chunk := response.read(_CHUNK):
                out.write(chunk)
                bar.update(len(chunk))


def download_uci_har(root: Path = DATA_ROOT, force: bool = False) -> Path:
    """Download and extract UCI HAR into ``root`` and verify it. Idempotent.

    Returns the path to the ``UCI HAR Dataset`` folder. An existing folder that verifies is
    kept unless ``force``. A new copy is extracted and verified in a staging folder first, so
    an existing copy is only replaced by one that passes verification.
    """
    root = Path(root)
    uci_dir = root / DATASET_DIRNAME
    if uci_dir.is_dir() and not force:
        try:
            verify_dataset(uci_dir)
            return uci_dir
        except (FileNotFoundError, ValueError) as exc:
            print(f"Existing copy failed verification, downloading again:\n{exc}", file=sys.stderr)

    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="_uci_har_", dir=root))
    try:
        archive = staging / "download.zip"
        errors = []
        for url in URLS:
            try:
                print(f"Downloading {url}")
                _fetch(url, archive)
                break
            except OSError as exc:  # URLError, HTTPError and socket timeouts are all OSErrors
                errors.append(f"{url}: {exc}")
        else:
            raise RuntimeError(
                "Could not download UCI HAR:\n  " + "\n  ".join(errors)
                + f"\nURL: {URLS[0]}\n{MANUAL_HINT}"
            )

        extracted = extract_archive(archive, staging / "extracted")
        verify_dataset(extracted)
        if uci_dir.exists():
            shutil.rmtree(uci_dir)
        shutil.move(str(extracted), str(uci_dir))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return uci_dir


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Download, extract and verify UCI HAR.")
    parser.add_argument("--root", type=Path, default=DATA_ROOT,
                        help="folder that will contain 'UCI HAR Dataset' (default: data/raw)")
    parser.add_argument("--force", action="store_true",
                        help="download again even if a verified copy exists")
    args = parser.parse_args(argv)
    try:
        path = download_uci_har(args.root, force=args.force)
    except (RuntimeError, FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"UCI HAR verified at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
