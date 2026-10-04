"""Tests for har.data.download: verification, nested-zip extraction and idempotent download."""

from __future__ import annotations

import urllib.error
import zipfile
from pathlib import Path

import pytest

import har.data.download as dl
from har.data.download import (
    DATASET_DIRNAME, MANUAL_HINT, URLS, download_uci_har, expected_shapes, extract_archive,
    text_matrix_shape, verify_dataset,
)
from har.utils.paths import UCI_DIR


def _make_tiny_dataset(parent: Path) -> Path:
    """A stand-in 'UCI HAR Dataset' folder with two small files (not a valid dataset)."""
    folder = parent / DATASET_DIRNAME
    (folder / "train").mkdir(parents=True)
    (folder / "activity_labels.txt").write_text("1 WALKING\n", encoding="utf-8")
    (folder / "train" / "y_train.txt").write_text("5\n5\n", encoding="utf-8")
    return folder


def _zip_folder(folder: Path, zip_path: Path, extra: dict[str, bytes] | None = None) -> Path:
    """Zip ``folder`` (stored under its own name) plus macOS junk entries and ``extra`` files."""
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in sorted(folder.rglob("*")):
            zf.write(path, path.relative_to(folder.parent).as_posix())
        zf.writestr(f"__MACOSX/{folder.name}/._activity_labels.txt", b"junk")
        zf.writestr(f"{folder.name}/.DS_Store", b"junk")
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return zip_path


def _assert_no_junk(root: Path) -> None:
    assert not any(p.name == "__MACOSX" for p in root.rglob("*"))
    assert not any(p.name == ".DS_Store" for p in root.rglob("*"))


# ---------------------------------------------------------------- verification

@pytest.mark.needs_data
def test_verify_dataset_accepts_the_real_folder() -> None:
    verify_dataset(UCI_DIR)


@pytest.mark.needs_data
def test_real_files_have_the_expected_shapes() -> None:
    assert text_matrix_shape(UCI_DIR / "train" / "X_train.txt") == (7352, 561)
    assert text_matrix_shape(UCI_DIR / "test" / "Inertial Signals" / "body_gyro_y_test.txt") == (2947, 128)


def test_expected_files_cover_all_raw_channels_and_feature_files() -> None:
    shapes = expected_shapes()
    for split, n in (("train", 7352), ("test", 2947)):
        assert shapes[f"{split}/X_{split}.txt"] == (n, 561)
        assert shapes[f"{split}/y_{split}.txt"] == (n, 1)
        assert shapes[f"{split}/subject_{split}.txt"] == (n, 1)
        signals = [k for k in shapes if k.startswith(f"{split}/Inertial Signals/")]
        assert len(signals) == 9
        assert all(shapes[k] == (n, 128) for k in signals)


def test_verify_dataset_raises_on_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        verify_dataset(tmp_path / "nowhere")


def test_verify_dataset_names_the_missing_files(tmp_path: Path) -> None:
    folder = _make_tiny_dataset(tmp_path)
    with pytest.raises(FileNotFoundError) as excinfo:
        verify_dataset(folder)
    message = str(excinfo.value)
    assert "train/X_train.txt" in message
    assert "test/Inertial Signals/total_acc_z_test.txt" in message
    assert "activity_labels.txt" not in message  # present files are not reported
    assert "train/y_train.txt" not in message


def test_verify_dataset_reports_wrong_shapes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Shrink the expectations so a complete but tiny fake dataset can be built quickly.
    folder = tmp_path / DATASET_DIRNAME
    small = {"a.txt": (2, 3), "sub/b.txt": (2, 1), "names.txt": (3, None)}
    monkeypatch.setattr(dl, "expected_shapes", lambda: small)
    (folder / "sub").mkdir(parents=True)
    (folder / "a.txt").write_text("1 2 3\n4 5 6\n", encoding="utf-8")
    (folder / "sub" / "b.txt").write_text("1\n2\n", encoding="utf-8")
    (folder / "names.txt").write_text("1 x\n2 y\n3 z\n", encoding="utf-8")
    verify_dataset(folder)  # all shapes match

    (folder / "a.txt").write_text("1 2 3 4\n5 6 7 8\n", encoding="utf-8")
    (folder / "sub" / "b.txt").write_text("1\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        verify_dataset(folder)
    message = str(excinfo.value)
    assert "a.txt: 2 x 4, expected 2 x 3" in message
    assert "sub/b.txt: 1 x 1, expected 2 x 1" in message


def test_text_matrix_shape_ignores_blank_lines_and_rejects_ragged_rows(tmp_path: Path) -> None:
    path = tmp_path / "m.txt"
    path.write_text("  1.0e-001  2.0\n3.0 4.0\n\n", encoding="utf-8")
    assert text_matrix_shape(path) == (2, 2)
    path.write_text("1 2\n3 4 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2 has 3 columns"):
        text_matrix_shape(path)


# ---------------------------------------------------------------- extraction

def test_extract_nested_zip_skips_junk(tmp_path: Path) -> None:
    """Current UCI layout: outer zip -> 'UCI HAR Dataset.names' + 'UCI HAR Dataset.zip' -> folder."""
    folder = _make_tiny_dataset(tmp_path / "src")
    inner = _zip_folder(folder, tmp_path / "inner.zip")
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.write(inner, f"{DATASET_DIRNAME}.zip")
        zf.writestr(f"{DATASET_DIRNAME}.names", b"description")
        zf.writestr("__MACOSX/._UCI HAR Dataset.zip", b"junk")
        zf.writestr(".DS_Store", b"junk")

    dest = tmp_path / "out"
    found = extract_archive(outer, dest)

    assert found == dest / DATASET_DIRNAME
    assert (found / "train" / "y_train.txt").read_text(encoding="utf-8") == "5\n5\n"
    assert not (dest / f"{DATASET_DIRNAME}.zip").exists()  # nested zip removed after extraction
    _assert_no_junk(dest)


def test_extract_flat_zip(tmp_path: Path) -> None:
    """Legacy UCI layout: the zip contains the folder directly."""
    folder = _make_tiny_dataset(tmp_path / "src")
    archive = _zip_folder(folder, tmp_path / "legacy.zip")
    found = extract_archive(archive, tmp_path / "out")
    assert found == tmp_path / "out" / DATASET_DIRNAME
    assert (found / "activity_labels.txt").is_file()
    _assert_no_junk(tmp_path / "out")


def test_extract_without_dataset_folder_raises(tmp_path: Path) -> None:
    archive = tmp_path / "other.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("something/else.txt", b"x")
    with pytest.raises(FileNotFoundError, match=DATASET_DIRNAME):
        extract_archive(archive, tmp_path / "out")


# ---------------------------------------------------------------- download

def _no_network(url: str, target: Path) -> None:
    raise AssertionError("download attempted")


def _offline(url: str, target: Path) -> None:
    raise urllib.error.URLError("offline")


def test_download_skips_a_verified_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = _make_tiny_dataset(tmp_path)
    monkeypatch.setattr(dl, "verify_dataset", lambda uci_dir: None)
    monkeypatch.setattr(dl, "_fetch", _no_network)
    assert download_uci_har(tmp_path) == folder


def test_download_extracts_verifies_and_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _zip_folder(_make_tiny_dataset(tmp_path / "src"), tmp_path / "legacy.zip")
    fetched = []

    def fake_fetch(url: str, target: Path) -> None:
        fetched.append(url)
        if url == URLS[0]:
            raise urllib.error.URLError("current URL unavailable")
        target.write_bytes(source.read_bytes())

    verified = []
    monkeypatch.setattr(dl, "_fetch", fake_fetch)
    monkeypatch.setattr(dl, "verify_dataset", lambda uci_dir: verified.append(Path(uci_dir)))
    root = tmp_path / "data"

    result = download_uci_har(root)

    assert fetched == list(URLS)  # fell back to the legacy URL
    assert result == root / DATASET_DIRNAME
    assert (result / "train" / "y_train.txt").is_file()
    assert verified, "the extracted copy must be verified before it is moved into place"
    assert [p.name for p in root.iterdir()] == [DATASET_DIRNAME]  # staging folder removed


def test_download_failure_explains_the_manual_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl, "_fetch", _offline)
    with pytest.raises(RuntimeError) as excinfo:
        download_uci_har(tmp_path)
    message = str(excinfo.value)
    assert URLS[0] in message
    assert MANUAL_HINT in message
    assert list(tmp_path.iterdir()) == []  # nothing left behind


def test_cli_returns_nonzero_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(dl, "_fetch", _offline)
    assert dl.main(["--root", str(tmp_path)]) == 1
    assert "in a browser" in capsys.readouterr().err
