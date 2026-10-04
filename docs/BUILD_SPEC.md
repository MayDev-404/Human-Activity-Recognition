# BUILD_SPEC.md: HAR Architecture Comparison, Interim (Phase 2) Build

This is the complete build specification for the ICT 4442 mini project, written for Claude Code and for the three team members who will run it. `CLAUDE.md` (repo root) holds the always-on rules; this file holds the detail.

**Project:** Comparative Evaluation of Deep Learning Architectures for Human Activity Recognition from Smartphone Inertial Sensor Signals
**Course:** ICT 4442 Deep Learning, School of Computer Engineering, MIT Manipal
**Phase:** 2, Interim Submission (Report Template Part B + GitHub repository with per-member commit history)

| ID | Member | Synopsis assignment |
|----|--------|---------------------|
| M1 | Mayank Kejariwal | Models 1 and 4; engineered-feature pipeline |
| M2 | Ishan Abhijit Saraf | Model 2; shared evaluation harness (metrics, confusion matrices, timing and parameter-count instrumentation) |
| M3 | Addyan Kumar | Model 3 (with GRU ablation); raw-signal loading, windowing, normalisation and the validation split |

Shared by all: literature review, error analysis, report. One repository, per-member commits. (Registration numbers are deliberately not in this file; see 10.4.)

---

## Contents

0. How to use this document
1. What the interim submission needs, and where each piece comes from
2. Timeline (one night plus one day) and the final-hour sequence
3. Repository layout and file ownership
4. Environment and Git identity setup (every member)
5. Shared contracts (interfaces every module must follow)
6. Tasks per member, with order and commit plans
7. Model specifications
8. Training, evaluation and results protocol
9. Literature review source material (verified)
10. Interim report assembly (Part B)
11. Contribution log and task status files
12. Risk and plan for remaining work (draft content)
13. Definition of done
14. Troubleshooting

---

## 0. How to use this document

### 0.1 Order of operations

1. **M1 (Mayank) goes first.** Creates the repo from this spec (task M1.0), pushes it to GitHub, adds Ishan and Addyan as collaborators. Nobody else can start coding until this is pushed.
2. **M2 and M3, while waiting:** set up Python, PyTorch, Git identity and `gh` (section 4), send Mayank your GitHub username, accept the invite, clone.
3. **Everyone:** open the cloned repo in Claude Code on **your own machine, logged into your own GitHub account**, paste your kickoff prompt from 0.2, and work through your tasks in section 6. Each member trains their own models on their own machine and commits their own results. That is what makes the commit history honest evidence of who did what.

### 0.2 Kickoff prompts (paste into Claude Code)

**M1, Mayank (in an empty folder containing only `CLAUDE.md`, `docs/BUILD_SPEC.md` and the report template docx):**
```
I am Mayank Kejariwal, member M1. Read CLAUDE.md and docs/BUILD_SPEC.md completely.
1. Show me my git user.name, user.email and `gh auth status`, and wait for my confirmation.
2. Do task M1.0 (repo scaffold) exactly as specified in section 6.
3. Ask me whether the repo should be public or private, create it with gh, push main,
   and give me the commands to add Ishan and Addyan as collaborators once I have their usernames.
Stop after M1.0 and summarise what exists.
```
Then, once Ishan and Addyan have cloned (no need to wait for M3.2; task M1.1 explains how to handle that dependency):
```
Continue with my remaining tasks from section 6 of docs/BUILD_SPEC.md, following the
"Order" line of my section, one branch and one PR per task. Run pytest before each push.
Ask me before any run longer than 10 minutes.
```

**M2, Ishan:**
```
I am Ishan Abhijit Saraf, member M2. Read CLAUDE.md and docs/BUILD_SPEC.md completely.
Show me my git user.name, user.email and `gh auth status`, and wait for my confirmation.
Then do my tasks from section 6 following the "Order" line of my section, one branch and
one PR per task, committing as I work. Run pytest before each push. M2.4 (train script)
blocks everyone's training runs, so merge it as soon as it passes. Do not edit files owned
by other members. Ask me before any run longer than 10 minutes.
```

**M3, Addyan:**
```
I am Addyan Kumar, member M3. Read CLAUDE.md and docs/BUILD_SPEC.md completely.
Show me my git user.name, user.email and `gh auth status`, and wait for my confirmation.
Then do my tasks from section 6 following the "Order" line of my section, one branch and
one PR per task, committing as I work. M3.2 (validation split) blocks Mayank, so merge it
as soon as it passes. Run pytest before each push. Do not edit files owned by other
members. Ask me before any run longer than 10 minutes.
```

### 0.3 Rules Claude Code must follow while using this spec

All rules in `CLAUDE.md` apply. In short: commit only as the local user, never spoof authors or dates, never rewrite pushed history, never invent numbers or references, never tune on the test set, stay inside your member's files.

---

## 1. What the interim submission needs, and where each piece comes from

Requirements from the Phase 2 guidelines and Report Template Part B (report 3 to 5 pages plus GitHub link):

| Requirement | Repo artifact | Owner |
|---|---|---|
| Completed literature review, 8 to 10 papers in a table (paper / method / dataset / key result / relevance) | `docs/lit/mayank.md` (4), `docs/lit/ishan.md` (5), `docs/lit/addyan.md` (4) = 13 papers; synthesis in `report/interim/sections/lit_summary.md` | All; synthesis M3 |
| Dataset acquisition and preprocessing pipeline (code + description) | `src/har/data/*`, `configs/split.json`, `scripts/data_report.py`, `report/interim/sections/preprocessing_raw.md`, `preprocessing_features.md`, `evaluation_protocol.md` | M3 (download, raw, split), M1 (features), M2 (protocol text) |
| At least 2 of the proposed models implemented with preliminary results | MLP (M1), 1D-CNN (M2), BiLSTM + GRU ablation (M3) all trained; Transformer (M1) implemented and tested, trained if time allows. Results in `results/<model>/<run_id>/`; observations in `report/interim/sections/observations.md` | M1, M2, M3; observations M2 |
| Individual contribution log to date (commit history + task matrix signed by all) | `docs/contribution_log.md` (generated from git), `docs/tasks/<member>.md`, signature table in the report | M3 (script), each member (own task file), all (sign) |
| Risk / plan for remaining models and timeline to Final | `report/interim/sections/risk_plan.md` | M1 (skeleton and plan), M2 and M3 (own model risks) |
| Report in Part B format | `report/build_interim_report.py` fills the official template into `report/build/ICT4442_Interim_Report_HAR.docx` (+ PDF) | M1 |

The synopsis work plan had Models 1, 2 and 3 done by the interim and Model 4 by 12 Oct. This build follows that plan: three models trained (one per member, which is also the strongest commit evidence), Model 4 implemented and tested, trained in Phase 3 unless there is spare time.

---

## 2. Timeline (one night plus one day) and the final-hour sequence

T0 = the moment M1's scaffold is pushed. Times are targets; dependencies matter more than clock times.

| Block | M1 Mayank | M2 Ishan | M3 Addyan |
|---|---|---|---|
| **Tonight, before T0 (30 min)** | M1.0 scaffold, create repo, invite collaborators | Section 4 setup, send GitHub username, clone after invite | Section 4 setup, send GitHub username, clone after invite |
| **Tonight, T0 to T0+1.5h** | M1.1 feature loader (merge after M3.2 lands) | M2.1 metrics, M2.2 profiling, start M2.3 trainer | M3.1 download, **M3.2 split (merged within ~45 min)**, start M3.3 raw loader |
| **Tonight, T0+1.5h to T0+3h** | M1.2 MLP code + tests; M1.3 Transformer code + tests; start M1.5 builder against dummy inputs | M2.3 trainer, **M2.4 train script (merge ASAP)** | M3.3 raw loader + normalisation merged; M3.6 BiRNN code + tests |
| **Tomorrow morning** | Train MLP, commit run; M1.4 lit rows | M2.5 CNN: train, commit run; M2.6 results table and plots | M3.6 train BiLSTM and GRU, commit runs; M3.4 augmentation; M3.5 data report |
| **Tomorrow midday** | M1.5 builder on real inputs; risk plan; (stretch) Transformer run | M2.7 lit rows; M2.8 protocol section + risk bullets; observations | M3.8 lit rows + synthesis; M3.9 preprocessing section + risk bullets; M3.7 log script |
| **Final 2 to 3 hours** | Final-hour sequence below | | |

What blocks what:

```
M1.0 scaffold ──> everything
M3.1 download ──> any code that reads real data
M3.2 split ─────> M1.1 feature loaders, M3.3 raw loaders
M2.1 to M2.4 harness ──> every training run (M1.2, M2.5, M3.6)
M1.5 report builder <── all results, lit rows, sections, contribution log
```

Models can be written and unit-tested **before** the data and harness are ready by using `make_synthetic_bundle()` (5.2), so nobody has to sit idle.

**Training on CPU is fine.** In a dry run of exactly these model definitions and this protocol on a weak 2-core cloud CPU, total training took about 15 s (MLP), about 2 minutes each (1D-CNN, BiLSTM) and about 10 minutes (Transformer). Laptops with more cores, or a GPU, are faster. If a member has no usable machine, train on Google Colab (set `git config user.name` there to your own name first), download the run folder, and commit it from your own machine and account.

### 2.1 Final-hour sequence (do these in order)

1. Every member's last full training runs are merged.
2. **M2:** run `make_results_table.py` and `plot_curves.py`; commit (`exp: regenerate results summary`). This also refreshes the README results block.
3. **Each member:** update the statuses in your own `docs/tasks/<member>.md`; commit.
4. **M3:** run `contribution_log.py`; commit (`docs: regenerate contribution log`). The file header records the HEAD sha it describes.
5. **M1:** pull, run `report/build_interim_report.py`, open the docx, check page count and that every number matches `results/summary.csv`.
6. **All three:** read the PDF; M2 double-checks the numbers; everyone signs the contribution table on the printed or PDF copy.
7. **M1:** tag and push: `git tag -a interim-v1 -m "Phase 2 interim submission" && git push origin interim-v1`. Submit the report and the repo link.

---

## 3. Repository layout and file ownership

Repo name: `har-dl-comparison`.

```
har-dl-comparison/
├── CLAUDE.md                              M1 (scaffold)
├── README.md                              M1; results block rewritten by M2's script
├── pyproject.toml                         M1
├── requirements.txt                       M1
├── .gitignore                             M1
├── .mailmap                               only if needed; each member adds lines for their OWN identities
├── .github/workflows/tests.yml            M1
├── configs/
│   ├── base.yaml                          M1  (protocol; changes need all 3)
│   ├── team.yaml                          M1  (member ids, names, GitHub usernames)
│   ├── split.json                         M3  (generated once, committed)
│   ├── mlp.yaml                           M1
│   ├── transformer.yaml                   M1
│   ├── cnn1d.yaml                         M2
│   ├── bilstm.yaml                        M3
│   └── gru.yaml                           M3
├── src/har/
│   ├── __init__.py                        M1
│   ├── utils/  seed.py config.py paths.py build.py        M1
│   ├── data/
│   │   ├── constants.py                   M1  (CHANNELS, ACTIVITY_NAMES, N_CLASSES, WINDOW_LEN)
│   │   ├── bundle.py                      M1  (DataBundle + synthetic bundle)
│   │   ├── features.py                    M1
│   │   ├── download.py                    M3
│   │   ├── raw.py                         M3
│   │   ├── splits.py                      M3
│   │   ├── normalize.py                   M3
│   │   └── augment.py                     M3
│   ├── models/
│   │   ├── mlp.py                         M1
│   │   ├── transformer.py                 M1
│   │   ├── cnn1d.py                       M2
│   │   └── birnn.py                       M3
│   ├── train/  trainer.py                 M2
│   └── eval/   metrics.py profiling.py plots.py           M2
├── scripts/
│   ├── train.py                           M2
│   ├── make_results_table.py              M2
│   ├── plot_curves.py                     M2
│   ├── profile_all.py                     M2
│   ├── data_report.py                     M3
│   └── contribution_log.py                M3
├── tests/
│   ├── conftest.py                        M1
│   ├── test_utils.py test_bundle.py test_features.py test_mlp.py test_transformer.py           M1
│   ├── test_metrics.py test_profiling.py test_trainer.py test_train_script.py test_cnn1d.py    M2
│   └── test_download.py test_splits.py test_raw.py test_normalize.py test_augment.py test_birnn.py   M3
├── results/                               each member commits their own run folders
│   ├── <model>/<run_id>/...               (run_by recorded inside metrics.json)
│   ├── summary.csv  summary.md            M2 (generated)
│   ├── profile_<label>.csv                M2 (generated by profile_all.py)
│   └── figures/                           M2 (generated)
├── docs/
│   ├── BUILD_SPEC.md                      M1 (scaffold)
│   ├── tasks/  mayank.md (M1)  ishan.md (M2)  addyan.md (M3)    created pre-filled by M1, then owner-only edits
│   ├── contribution_log.md                M3 (generated)
│   ├── dataset_stats.md                   M3 (generated)
│   ├── figures/                           M3 (generated by data_report.py)
│   └── lit/  README.md (M1) mayank.md (M1) ishan.md (M2) addyan.md (M3)
├── report/
│   ├── template/ICT_4442_Mini_Project_Report_Template.docx    M1
│   ├── build_interim_report.py            M1
│   ├── interim/
│   │   ├── meta.yaml                      M1
│   │   ├── members.local.yaml             registration numbers, on M1's machine only (gitignored)
│   │   └── sections/
│   │       ├── lit_summary.md             M3
│   │       ├── preprocessing_raw.md       M3
│   │       ├── preprocessing_features.md  M1
│   │       ├── evaluation_protocol.md     M2
│   │       ├── model_mlp.md model_transformer.md   M1
│   │       ├── model_cnn1d.md             M2
│   │       ├── model_bilstm.md            M3  (covers the GRU ablation too)
│   │       ├── observations.md            M2
│   │       └── risk_plan.md               M1 creates in scaffold; M2 and M3 add bullets under their own headings
│   └── build/                             gitignored (contains registration numbers); see 10.4
├── notebooks/                             optional exploration only
└── data/                                  gitignored (dataset lives here)
```

Shared files (`README.md`, `risk_plan.md`) get small edits from more than one member. Always `git pull` right before editing them and touch only your own section.

---

## 4. Environment and Git identity setup (every member, ~20 minutes)

### 4.1 Python and PyTorch

```bash
# Python 3.10 to 3.13
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch **before** the requirements file, otherwise pip may install the CPU build first and then report the CUDA build as "already satisfied":
- **NVIDIA GPU (Mayank's laptop):** the exact command from https://pytorch.org/get-started/locally/ for your CUDA version (`nvidia-smi` shows it). Check: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
- **CPU only:** `pip install torch`.
- Then, after cloning: `pip install -r requirements.txt && pip install -e .`

### 4.2 Git identity (critical for commit verification)

GitHub only credits a commit to your profile if the commit email is one of the **verified emails on your GitHub account** (or your GitHub noreply address).

```bash
git config --global user.name  "Your Full Name"            # e.g. "Ishan Abhijit Saraf"
git config --global user.email "email-on-your-github-account"
git config --global user.name ; git config --global user.email   # check
```

The noreply address is shown under GitHub, Settings, Emails ("Keep my email addresses private"): `<id>+<username>@users.noreply.github.com`. Using it keeps your personal email out of public history.

After your first push, open the commit on GitHub and confirm your avatar and username appear next to it. A grey unknown avatar means the email does not match your account; see section 14.

### 4.3 GitHub CLI

Install `gh` from https://cli.github.com, then `gh auth login` with **your own** account (`gh auth status` to check). Claude Code uses it to open PRs, and PR authorship is part of the evidence.

### 4.4 Joining the repo (M2, M3)

1. Send Mayank your GitHub username.
2. Accept the invite (email or https://github.com/notifications).
3. `git clone https://github.com/<mayank-username>/har-dl-comparison.git && cd har-dl-comparison`
4. Activate the venv, install (4.1), `pytest -q`.

---

## 5. Shared contracts (interfaces every module must follow)

These signatures are fixed so all three members can work in parallel. If a contract must change, the owner changes it in a dedicated PR and tells the others.

### 5.1 `har/data/constants.py` (M1, scaffold; imported by everyone)

```python
CHANNELS: list[str] = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
ACTIVITY_NAMES: list[str] = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING",
]
N_CLASSES = 6
WINDOW_LEN = 128   # 2.56 s at 50 Hz, 50% overlap (pre-segmented by the dataset authors)
N_FEATURES = 561
```

### 5.2 `har/data/bundle.py` (M1, scaffold)

```python
@dataclass
class DataBundle:
    train: DataLoader          # shuffled with a seeded torch.Generator
    val: DataLoader            # not shuffled
    test: DataLoader           # not shuffled
    input_shape: tuple[int, ...]   # (561,) for features, (128, 9) for raw
    n_classes: int             # 6
    class_names: list[str]     # ACTIVITY_NAMES
    meta: dict                 # representation, counts per split, val_subjects, normalisation info

def make_synthetic_bundle(input_shape: tuple[int, ...], n_train: int = 256, n_val: int = 64,
                          n_test: int = 64, n_classes: int = 6, batch_size: int = 32,
                          seed: int = 0, learnable: bool = True) -> DataBundle:
    """Random data for unit tests. If learnable, labels depend on the inputs
    (e.g. argmax of a fixed random projection of the flattened input) so a model can fit them."""
```

Every DataLoader yields `(x: float32 tensor, y: int64 tensor)` with `x.shape == (B, *input_shape)`. DataLoaders use `num_workers=0`.

### 5.3 Data loaders

```python
# har/data/download.py (M3)
def download_uci_har(root: Path = DATA_ROOT, force: bool = False) -> Path:
    """Idempotent. Returns the path to the 'UCI HAR Dataset' folder."""
def verify_dataset(uci_dir: Path = UCI_DIR) -> None:
    """Raises with a clear message if any expected file is missing or has the wrong shape."""
# CLI: python -m har.data.download

# har/data/splits.py (M3)
N_VAL_SUBJECTS = 6
SPLIT_SEED = 42
def select_val_subjects(train_subjects: Sequence[int], n: int = N_VAL_SUBJECTS,
                        seed: int = SPLIT_SEED) -> list[int]
def load_val_subjects(split_file: Path = SPLIT_FILE) -> list[int]
def train_val_masks(subjects: np.ndarray, val_subjects: Sequence[int]) -> tuple[np.ndarray, np.ndarray]
# CLI: python -m har.data.splits --write   (writes configs/split.json once)

# har/data/raw.py (M3)
def load_raw_split(split: Literal["train", "test"], uci_dir: Path = UCI_DIR
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X float32 (N, 128, 9) in CHANNELS order, y int64 (N,) in 0..5, subjects int64 (N,)."""
def build_raw_loaders(batch_size: int, seed: int, augment: bool = False,
                      uci_dir: Path = UCI_DIR, split_file: Path = SPLIT_FILE,
                      subset: int | None = None) -> DataBundle

# har/data/normalize.py (M3)
class ChannelStandardizer:
    def fit(self, x: np.ndarray) -> "ChannelStandardizer"   # x (N, T, C); per-channel mean/std over N and T
    def transform(self, x: np.ndarray) -> np.ndarray
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> "ChannelStandardizer"

# har/data/features.py (M1)
def load_feature_split(split: Literal["train", "test"], uci_dir: Path = UCI_DIR
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X float32 (N, 561), y int64 (N,) in 0..5, subjects int64 (N,)."""
def feature_names(uci_dir: Path = UCI_DIR) -> list[str]:
    """561 names from features.txt, made unique (the file has duplicates)."""
def build_feature_loaders(batch_size: int, seed: int, uci_dir: Path = UCI_DIR,
                          split_file: Path = SPLIT_FILE, subset: int | None = None) -> DataBundle
```

`subset` (smoke tests only) keeps the first `subset` training windows after the split.

### 5.4 Models

Every model:
- is an `nn.Module` whose constructor takes only keyword arguments with defaults, matching `configs/<model>.yaml` `model.params`;
- `forward(x)` takes `(B, *input_shape)` and returns **logits** `(B, 6)` (no softmax);
- handles its own layout changes internally (for example the CNN permutes `(B, 128, 9)` to `(B, 9, 128)`).

Models are built from config with `har.utils.build.build_from_target(cfg["model"]["target"], cfg["model"]["params"])`, so no shared registry file needs editing.

### 5.5 Harness (M2)

```python
# har/eval/metrics.py
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """{'accuracy', 'macro_f1', 'per_class': {name: {'precision','recall','f1','support'}},
        'confusion': list[list[int]]}   # always 6 classes, labels=list(range(6)), zero_division=0"""
def classification_report_text(y_true, y_pred, class_names) -> str

# har/eval/plots.py
def plot_confusion(cm, class_names, path: Path, normalize: bool = True, title: str = "") -> None
def plot_history(history: list[dict], path: Path, title: str = "") -> None   # loss + val macro-F1 curves

# har/eval/profiling.py
def count_params(model: nn.Module, trainable_only: bool = True) -> int
def model_size_mb(model: nn.Module) -> float          # float32 parameters + buffers
def measure_latency(model: nn.Module, input_shape: tuple[int, ...], device: str = "cpu",
                    batch_size: int = 1, warmup: int = 20, iters: int = 200,
                    threads: int | None = 1) -> dict   # {'mean_ms','p50_ms','p95_ms','batch_size','device','threads'}
def hardware_info() -> dict                          # python, torch, os, cpu, device, gpu name

# har/train/trainer.py
@dataclass
class TrainConfig:
    batch_size: int; max_epochs: int; optimizer: str; lr: float; weight_decay: float
    patience: int; monitor: str; mode: str; grad_clip_norm: float; device: str
    # optimizer must be "adam"; anything else raises ValueError (protocol)
    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig"

@dataclass
class TrainResult:
    best_state: dict; best_epoch: int; epochs_run: int
    history: list[dict]            # one row per epoch, see 8.3
    train_time_s: float; mean_epoch_time_s: float

def resolve_device(name: str = "auto") -> torch.device
def fit(model: nn.Module, bundle: DataBundle, cfg: TrainConfig) -> TrainResult
def predict(model: nn.Module, loader: DataLoader, device: torch.device
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]   # y_true, y_pred, probs
```

### 5.6 Utilities (M1, scaffold)

```python
# har/utils/paths.py
REPO_ROOT: Path      # directory containing pyproject.toml
DATA_ROOT = REPO_ROOT / "data" / "raw"
UCI_DIR = DATA_ROOT / "UCI HAR Dataset"
CONFIGS = REPO_ROOT / "configs"
SPLIT_FILE = CONFIGS / "split.json"
RESULTS_ROOT = REPO_ROOT / "results"

# har/utils/seed.py
def set_seed(seed: int, deterministic: bool = True) -> None
    # random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all;
    # if deterministic: torch.backends.cudnn.deterministic = True, benchmark = False.
    # Do NOT call torch.use_deterministic_algorithms(True): it errors on CUDA without CUBLAS_WORKSPACE_CONFIG.
def make_generator(seed: int) -> torch.Generator

# har/utils/config.py
class ProtocolError(Exception): ...
def load_config(model_cfg: Path, base_cfg: Path = CONFIGS / "base.yaml",
                allow_protocol_override: bool = False) -> dict:
    """Deep-merge base + model config. Raise ProtocolError if the model config contains
    top-level 'train' or 'seed' keys and allow_protocol_override is False.
    The resolved dict records 'protocol_override': bool."""

# har/utils/build.py
def build_from_target(target: str, params: dict) -> Any   # importlib "pkg.module.ClassName"
```

### 5.7 `configs/base.yaml` (M1, scaffold)

```yaml
seed: 42
data:
  augment: false
train:
  batch_size: 64
  max_epochs: 50
  optimizer: adam
  lr: 0.001
  weight_decay: 0.0
  patience: 10
  monitor: val_macro_f1
  mode: max
  grad_clip_norm: 1.0
  device: auto
eval:
  latency_batch_sizes: [1, 256]
  latency_warmup: 20
  latency_iters: 200
  latency_threads: 1
```

Model configs contain only `name`, `owner`, `data.representation` (and optionally `data.augment`, which stays `false` for comparison runs) and `model`.

### 5.8 `configs/team.yaml` (M1, scaffold)

```yaml
members:
  - {id: M1, name: "Mayank Kejariwal",    github: ""}
  - {id: M2, name: "Ishan Abhijit Saraf", github: ""}
  - {id: M3, name: "Addyan Kumar",        github: ""}
```
GitHub usernames are filled in when known. No emails or registration numbers here.

---

## 6. Tasks per member, with order and commit plans

Each task lists: branch, files, steps, acceptance criteria, and indicative commits (commit as you actually work; see CLAUDE.md section 2). In the PR that completes a task, also update that task's row in your own `docs/tasks/<member>.md`.

### M1: Mayank Kejariwal

**Order:** M1.0 → M1.1 → M1.2 (code and tests) → M1.3 (code and tests) → M1.5 part 1 (builder against dummy inputs) → M1.2 training run (after M2.4 is merged) → M1.4 → M1.5 part 2 (real inputs, risk plan) → M1.3 stretch run if time.

#### M1.0 Repository scaffold (directly on `main`; the first commits of the repo)

Steps:
1. `git init -b main` in the folder that holds `CLAUDE.md`, `docs/BUILD_SPEC.md` and the template docx.
2. Create:
   - `.gitignore`: standard Python entries **except `build/`** (it would also hide paths you may want later), plus `/data/` (leading slash, so `src/har/data/` is NOT ignored), `*.pt`, `*.pth`, `.venv/`, `results/_smoke/`, `*.local.yaml`, `report/build/`, `.DS_Store`, `__pycache__/`, `.ipynb_checkpoints/`, `~$*`. Afterwards check with `git check-ignore -v src/har/data/bundle.py` (must print nothing).
   - `pyproject.toml`: project `har`, version 0.1.0, `requires-python = ">=3.10"`, src layout (`[tool.setuptools.packages.find] where = ["src"]`), pytest config (`testpaths = ["tests"]`, marker `needs_data`).
   - `requirements.txt`: `torch>=2.1`, `numpy>=1.24`, `pandas>=2.0`, `scikit-learn>=1.3`, `matplotlib>=3.7`, `pyyaml>=6.0`, `tqdm>=4.66`, `python-docx>=1.1`, `pypdf>=4.0`, `pytest>=7.4`.
   - `README.md`: title, team and ownership table (names and GitHub usernames only), dataset and licence note (CC BY 4.0; cite the dataset and Anguita et al. 2013, see 9.2), setup, how to run, a "Results" section containing the markers `<!-- RESULTS:START -->` and `<!-- RESULTS:END -->` (M2's script writes between them), and an **AI assistance** section: "Code in this repository was developed with the help of Claude Code (Anthropic). Commits made with it carry a Co-Authored-By trailer. Each member reviewed, ran and is responsible for the modules they own."
   - `configs/base.yaml` (5.7) and `configs/team.yaml` (5.8).
   - `src/har/__init__.py` and empty `__init__.py` in `utils/`, `data/`, `models/`, `train/`, `eval/`.
   - `src/har/utils/{paths,seed,config,build}.py` (5.6), `src/har/data/constants.py` (5.1), `src/har/data/bundle.py` (5.2).
   - `tests/conftest.py`: a `data_available()` helper and a collection hook so tests marked `needs_data` skip with "UCI HAR not downloaded; run python -m har.data.download".
   - `tests/test_utils.py` (config merge, ProtocolError on a `train` key, build_from_target, set_seed reproducibility) and `tests/test_bundle.py` (synthetic bundle shapes and dtypes, learnable labels).
   - `.github/workflows/tests.yml`: on push and pull_request, ubuntu-latest, Python 3.11, `pip install torch --index-url https://download.pytorch.org/whl/cpu`, `pip install -r requirements.txt && pip install -e .`, `pytest -q`. Data-dependent tests skip in CI.
   - `docs/tasks/mayank.md`, `docs/tasks/ishan.md`, `docs/tasks/addyan.md`, each pre-filled with that member's task IDs from this section, status `Todo` (format in 11.2). After this commit, only the owner edits each file.
   - `docs/lit/README.md` with the row format from 9.1.
   - `report/interim/sections/risk_plan.md` skeleton with headings `## Remaining work`, `## Risks: Model 1 and 4 (M1)`, `## Risks: Model 2 (M2)`, `## Risks: Model 3 (M3)`, `## Shared risks`, `## Timeline to Final`.
   - Move the template to `report/template/ICT_4442_Mini_Project_Report_Template.docx`.
3. Indicative commits: `chore: project structure, gitignore and packaging`; `docs: CLAUDE.md, build spec and report template`; `feat(utils): paths, seeding, protocol-locked config loader, build_from_target`; `feat(data): constants, DataBundle contract and synthetic bundle`; `test: utils and bundle tests`; `ci: run pytest on push and pull request`; `docs: task status files, literature format, risk plan skeleton`.
4. Ask the human: public or private repo. Public is simplest for evaluators; if private, the human must add the instructor as a collaborator. Then `gh repo create har-dl-comparison --public` (or `--private`) `--source . --push`, and allow only merge commits so individual commits survive every PR: `gh repo edit --enable-merge-commit --enable-squash-merge=false --enable-rebase-merge=false`.
5. Add collaborators once usernames arrive: `gh api -X PUT repos/<owner>/har-dl-comparison/collaborators/<username> -f permission=push` (or GitHub, Settings, Collaborators). Put the usernames into `configs/team.yaml` in a small follow-up commit.

Acceptance: fresh clone, install, `pytest -q` passes; CI green; `python -c "import har.data.bundle"` works; `git check-ignore` confirms `src/har/data/` is tracked.

#### M1.1 Engineered-feature pipeline (branch `mayank/features`)

Files: `src/har/data/features.py`, `tests/test_features.py`, `report/interim/sections/preprocessing_features.md`.

Steps:
1. `load_feature_split`: read `X_<split>.txt` (whitespace separated: `pd.read_csv(path, sep=r"\s+", header=None)`), `y_<split>.txt`, `subject_<split>.txt` from `UCI_DIR/<split>/`. Validate: train (7352, 561), test (2947, 561), no NaN or inf, labels in 1..6 (then subtract 1), label and subject files have the same row count as X.
2. `feature_names()`: read `features.txt`. It contains **duplicate names** (561 names but only 477 unique, from the `bandsEnergy` features), so append the column index when a name repeats.
3. `build_feature_loaders`: masks from `har.data.splits.train_val_masks(subjects_train, load_val_subjects())`; fit `sklearn.preprocessing.StandardScaler` on **train-mask rows only**; transform train, val and test; wrap in `TensorDataset`; train loader shuffled with `make_generator(seed)`. `meta`: representation `features`, counts, val_subjects, `normalisation: "StandardScaler fitted on 15 training subjects"`. (The dataset authors already scaled features to [-1, 1]; standardisation is still applied so all models see zero-mean unit-variance inputs fitted without leakage.)
4. Tests (`needs_data` where real data is used): shapes and dtypes; label range 0..5; train and val subject sets disjoint and val equals `configs/split.json`; scaled train-split means are approximately 0 while val means are not forced to 0 (proves the fit used train only); train + val rows = 7352. **Alignment test:** using `pytest.importorskip("har.data.raw")`, check that `load_raw_split` and `load_feature_split` return identical `y` and `subjects` arrays for both splits (the test skips until M3.3 is merged).
5. Write `preprocessing_features.md` (budget in 10.3).

Dependency: needs `har.data.splits` from M3.2. Write the code first; once M3.2 is merged, `git merge main` into your branch, run the tests, then open the PR.

Indicative commits: `feat(data): engineered-feature loading with validation checks`; `feat(data): feature DataLoaders with train-only standardisation`; `test(data): feature pipeline and raw/feature alignment tests`; `docs(report): engineered-feature preprocessing section`.

#### M1.2 Model 1, MLP (branch `mayank/mlp`)

Files: `src/har/models/mlp.py`, `configs/mlp.yaml`, `tests/test_mlp.py`, `report/interim/sections/model_mlp.md`, then `results/mlp/<run_id>/`.

Architecture and config: 7.1. Tests: output shape `(B, 6)` on `make_synthetic_bundle((561,))`; parameter count equals 7.1; model fits the learnable synthetic bundle (train accuracy clearly above chance after a few epochs); batch size 1 in eval mode.

After M2.4 is merged: `python scripts/train.py --config configs/mlp.yaml --smoke`, then the full run. Commit the run folder. Then write `model_mlp.md` (format in 10.3) from the actual result.

Indicative commits: `feat(models): MLP baseline on engineered features`; `exp(mlp): preliminary run on official split`; `docs(report): MLP model notes`.

#### M1.3 Model 4, Transformer encoder (branch `mayank/transformer`)

Files: `src/har/models/transformer.py`, `configs/transformer.yaml`, `tests/test_transformer.py`, `report/interim/sections/model_transformer.md`.

Architecture: 7.4. Tonight: implementation plus tests (shape, parameter count, fits synthetic data, batch size 1). **Stretch:** a preliminary run if time allows (fast on the laptop GPU); otherwise status "Implemented and unit-tested; training scheduled for week of 5 Oct".

Indicative commits: `feat(models): Transformer encoder with learned positional embedding`; optionally `exp(transformer): preliminary run`; `docs(report): Transformer status`.

#### M1.4 Literature review rows (branch `mayank/lit`)

File: `docs/lit/mayank.md` with the 4 papers allocated to M1 in 9.2, in your own words. Commit: `docs(lit): rows for feature-based and attention-based HAR`.

#### M1.5 Report assembly and risk plan (branch `mayank/report`)

Files: `report/build_interim_report.py`, `report/interim/meta.yaml`, `report/interim/sections/risk_plan.md` (plan and M1 risks), `report/interim/members.local.yaml` (local only).

- Part 1 (tonight): write the builder (section 10) and run it against dummy inputs kept in a temporary folder outside git. Commit the builder code, not the dummy inputs.
- Part 2 (tomorrow midday): fill `meta.yaml`, write the "Remaining work", "Risks: Model 1 and 4" and "Timeline to Final" parts of `risk_plan.md` (section 12), and build on real inputs. The final build happens in step 5 of 2.1.

Indicative commits: `feat(report): Part B report builder`; `docs(report): risk plan and timeline`; `fix(report): ...` as needed.

---

### M2: Ishan Abhijit Saraf

**Order:** M2.1 → M2.2 → M2.3 → M2.4 → M2.5 → M2.6 → M2.7 → M2.8.

#### M2.1 Metrics and plots (branch `ishan/metrics`)

Files: `src/har/eval/metrics.py`, `src/har/eval/plots.py`, `tests/test_metrics.py`.

Use `sklearn.metrics`, always with `labels=list(range(6))` and `zero_division=0`: `accuracy_score`, `f1_score(average="macro")`, `precision_recall_fscore_support`, `confusion_matrix`. Without `labels`, a class missing from a batch shifts the per-class arrays and silently mislabels them. `plot_confusion` saves a PNG with class names on both axes, row-normalised percentages plus counts, matplotlib Agg backend, dpi 200. `plot_history` saves train/val loss and val macro-F1 against epoch.

Tests: a hand-computed toy example (known accuracy and macro-F1); confusion matrix is 6x6 and per-class dict has 6 entries even if a class is absent; output is JSON-serialisable; plots write files.

Indicative commits: `feat(eval): accuracy, macro-F1, per-class metrics and confusion matrix`; `feat(eval): confusion matrix and training curve plots`.

#### M2.2 Profiling (branch `ishan/profiling`)

Files: `src/har/eval/profiling.py`, `tests/test_profiling.py`.

- `count_params`: sum of `p.numel()` over trainable parameters.
- `measure_latency`: work on a `copy.deepcopy` of the model moved to the target device; `eval()`, `torch.inference_mode()`, random input `(batch_size, *input_shape)`, `warmup` untimed passes, then `iters` timed passes with `time.perf_counter()`. Synchronise before reading the clock: `torch.cuda.synchronize()` on CUDA, `torch.mps.synchronize()` on MPS. On CPU temporarily `torch.set_num_threads(threads)` and restore afterwards. Single-thread batch-1 CPU latency is the on-device proxy used in the comparison.
- `hardware_info`: Python, torch, OS, `platform.processor()`, CUDA device name if any.

Tests: `nn.Linear(10, 5)` has 55 params; latency dict values are positive; thread count restored after the call.

Indicative commits: `feat(eval): parameter count, model size and latency profiling`.

#### M2.3 Trainer (branch `ishan/trainer`)

Files: `src/har/train/trainer.py`, `tests/test_trainer.py`.

Behaviour (protocol in 8.1):
- `resolve_device("auto")`: cuda, then mps, then cpu.
- Each epoch: train pass (CrossEntropyLoss, Adam, `clip_grad_norm_(model.parameters(), grad_clip_norm)`), then a validation pass in `model.eval()` under `torch.inference_mode()` computing val loss, accuracy and macro-F1.
- History row per epoch (8.3).
- Early stopping on `val_macro_f1` (mode max, patience 10). Keep a deep copy of the best `state_dict`; restore it before returning.
- tqdm progress bar plus one summary line per epoch.

Tests (synthetic bundle, CPU, a few seconds): loss decreases; early stopping triggers with small patience; restored weights reproduce the best epoch's val score; `TrainConfig.from_dict(base["train"])` works and rejects a non-adam optimizer.

Indicative commits: `feat(train): training loop with early stopping on validation macro-F1`.

#### M2.4 Training script (branch `ishan/train-script`; merge as early as possible)

Files: `scripts/train.py`, `tests/test_train_script.py`.

`train.py` exposes `main(argv: list[str] | None = None) -> Path` (returns the run folder) so it can be tested.

CLI: `--config PATH` (required); `--smoke` (subset 512 training windows, 2 epochs, latency with warmup 2, iters 5, batch size 1 only, output under `results/_smoke/`); `--no-test` (skip test evaluation; use while debugging); `--allow-protocol-override` (Phase 3 tuning only; recorded); `--tag TEXT` (appended to run_id); `--output-root PATH` (default `results/`, used by tests).

Flow:
1. `cfg = load_config(...)`; `set_seed(cfg["seed"])`.
2. Data by `cfg["data"]["representation"]`: `"features"` calls `har.data.features.build_feature_loaders`; `"raw"` calls `har.data.raw.build_raw_loaders`; `"synthetic"` calls `make_synthetic_bundle(tuple(cfg["data"]["input_shape"]))` (tests only). Import lazily so a not-yet-merged module gives a clear error.
3. `model = build_from_target(...)`; `fit(...)`.
4. Evaluate the restored best model on val and (unless `--no-test`) test with `predict` and `compute_metrics`.
5. Profile: params, size, CPU latency at the batch sizes in `eval.latency_batch_sizes` with `latency_threads`, plus device latency at batch 1 on CUDA or MPS.
6. Write the run folder (8.2) and print a one-line summary.

Run identity: `run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{git_short_sha}"` (+ `_tag`); `git_dirty = bool(git status --porcelain)`; `run_by` from `git config user.name`, falling back to `"unknown (git user.name not set)"` with a warning if git is missing or the name is empty.

Test: `tests/test_train_script.py` writes a temporary config with `data: {representation: synthetic, input_shape: [16]}` and `model: {target: torch.nn.Linear, params: {in_features: 16, out_features: 6}}`, calls `main([... "--smoke", "--output-root", str(tmp_path)])`, and asserts that the run folder contains every file in 8.2 and that `metrics.json` has every key in the schema.

Acceptance: the test passes; a smoke run on real data works for `features` and `raw` once M1.1 and M3.3 are merged.

Indicative commits: `feat(scripts): end-to-end train script with evaluation, profiling and run folders`; `test(scripts): train script test on synthetic data`.

#### M2.5 Model 2, 1D-CNN (branch `ishan/cnn1d`)

Files: `src/har/models/cnn1d.py`, `configs/cnn1d.yaml`, `tests/test_cnn1d.py`, `report/interim/sections/model_cnn1d.md`, then `results/cnn1d/<run_id>/`.

Architecture: 7.2. Tests: shape, parameter count, fits synthetic raw-shaped data, batch size 1 in eval mode. Smoke run, full run, commit the run folder, write `model_cnn1d.md` from the actual result.

Indicative commits: `feat(models): 1D-CNN on raw inertial windows`; `exp(cnn1d): preliminary run on official split`; `docs(report): 1D-CNN model notes`.

#### M2.6 Results table, curves, same-machine profiling, observations (branch `ishan/results-table`)

Files: `scripts/make_results_table.py`, `scripts/plot_curves.py`, `scripts/profile_all.py`, `report/interim/sections/observations.md`.

- `make_results_table.py`: iterate over model names from `configs/*.yaml` (excluding `base.yaml` and `team.yaml`), so non-model folders such as `results/figures/` are never touched. For each model, consider all full runs (exclude `_smoke` and runs with `protocol_override: true`) and **select the run with the best validation macro-F1**. Write `results/summary.csv` (model key + raw floats, no formatting) and `results/summary.md` (formatted: accuracy as %, F1 to 3 decimals) with columns: Model, Run by, Representation, Params, Val Acc, Val Macro-F1, Test Acc, Test Macro-F1, Best epoch, Train time (s), CPU latency bs=1 (ms), Hardware, Runs (number of full runs), Run ID. Then rewrite the README block between `<!-- RESULTS:START -->` and `<!-- RESULTS:END -->` with `summary.md`.
- `plot_curves.py`: overlay validation macro-F1 per epoch for the selected runs into `results/figures/val_macro_f1_curves.png`, and copy each selected run's `confusion_test.png` to `results/figures/confusion_test_<model>.png`.
- `profile_all.py --label <machine-label>`: instantiate every model config **without training** (latency and parameter count do not depend on weights), measure on the current machine, write `results/profile_<label>.csv`. Use a neutral label like `mayank-laptop`, not the hostname. In Phase 3 this runs on one machine for the final cost comparison, because members' laptops differ.
- `observations.md` (after all interim runs are in): 3 to 5 sentences drawn only from `results/summary.csv` and the confusion matrices, labelled preliminary (untuned, single seed, timings from different machines). Budget in 10.3.

Indicative commits: one per script; `exp: regenerate results summary` whenever new runs land; `docs(report): preliminary observations`.

#### M2.7 Literature review rows (branch `ishan/lit`)

File: `docs/lit/ishan.md` with the 5 papers allocated to M2 in 9.2. Commit: `docs(lit): rows for convolutional and hybrid HAR models`.

#### M2.8 Evaluation protocol section and CNN risks (branch `ishan/report-eval`)

Files: `report/interim/sections/evaluation_protocol.md` (10.3) and your bullets under `## Risks: Model 2 (M2)` in `risk_plan.md`. Commit: `docs(report): evaluation protocol section and CNN risks`.

---

### M3: Addyan Kumar

**Order:** M3.1 → M3.2 → M3.3 → M3.6 → M3.4 → M3.5 → M3.8 → M3.9 → M3.7 (and the final log regeneration in step 4 of 2.1).

#### M3.1 Dataset download and verification (branch `addyan/download`)

Files: `src/har/data/download.py`, `tests/test_download.py`.

Verified facts about the download (checked 2 Oct 2026):
- Current URL: `https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip` (about 61 MB). It contains `UCI HAR Dataset.names` and a **nested** `UCI HAR Dataset.zip`, which contains the `UCI HAR Dataset/` folder.
- Legacy URL still works: `https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip` (contains the folder directly).
- Both archives include `__MACOSX/` and `.DS_Store` entries; skip them.

Steps: download with `urllib.request` and a tqdm progress bar into `data/raw/`, trying the current URL then the legacy one; extract, and if a nested zip is found, extract it too; skip junk entries; call `verify_dataset`. `verify_dataset` checks `activity_labels.txt`, `features.txt`, `train/X_train.txt` (7352 x 561), `test/X_test.txt` (2947 x 561), `y_*` and `subject_*` row counts, and all 9 `Inertial Signals/<channel>_<split>.txt` files per split with 128 columns. Idempotent: if the folder exists and verifies, do nothing unless `force=True`. If the network fails, print: "Download the zip from the URL above in a browser and place the extracted 'UCI HAR Dataset' folder in data/raw/".

Tests: `verify_dataset` on the real folder (`needs_data`); `verify_dataset` raises on a temp folder with a missing file; nested-zip extraction works on a tiny zip built in a temp directory.

Indicative commits: `feat(data): UCI HAR download with nested-zip extraction and verification`; `test(data): download and verification tests`.

#### M3.2 Validation split (branch `addyan/split`; **merge first, M1 is waiting on it**)

Files: `src/har/data/splits.py`, `configs/split.json`, `tests/test_splits.py`.

Facts: train subjects are `1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30`; test subjects are `2, 4, 9, 10, 12, 13, 18, 20, 24`.

- `select_val_subjects`: `rng = np.random.default_rng(seed)`; `sorted(int(s) for s in rng.choice(sorted(set(train_subjects)), size=n, replace=False))`.
- `python -m har.data.splits --write` reads `subject_train.txt`, selects, and writes `configs/split.json`:
  ```json
  {"seed": 42, "n_val_subjects": 6, "val_subjects": [ ... ], "train_subjects": [ ... 15 ids ... ],
   "test_subjects": [2, 4, 9, 10, 12, 13, 18, 20, 24], "note": "Selected once; do not regenerate."}
  ```
- `train_val_masks`: boolean masks over windows.

Cross-check (computed on the real data with current NumPy while writing this spec): the selection gives `val_subjects = [3, 15, 19, 22, 28, 29]`, leaving `train_subjects = [1, 5, 6, 7, 8, 11, 14, 16, 17, 21, 23, 25, 26, 27, 30]`, which splits the windows into **5,276 train / 2,076 val / 2,947 test**. If your output differs, check that you sorted the unique subject IDs before calling `choice`. Whatever the script produces, the committed `split.json` is the source of truth from then on.

Tests: 6 val subjects, all within the 21 training subjects; train, val and test subject sets pairwise disjoint; `select_val_subjects` is deterministic and equals the committed `split.json`; masks cover every training window exactly once.

Indicative commits: `feat(data): subject-wise validation split selection`; `chore(data): commit fixed validation split (seed 42)`.

#### M3.3 Raw-signal loader and normalisation (branch `addyan/raw-loader`)

Files: `src/har/data/raw.py`, `src/har/data/normalize.py`, `tests/test_raw.py`, `tests/test_normalize.py`.

- `load_raw_split`: import `CHANNELS` from `har.data.constants`; for each name in `CHANNELS` (explicit order, never directory-listing order) read `Inertial Signals/<name>_<split>.txt` as `(N, 128)` and stack on the last axis into `(N, 128, 9)` float32. Labels from `y_<split>.txt` minus 1; subjects from `subject_<split>.txt`. Check no NaN and that N matches the label file.
- "Windowing": the dataset is already segmented into 2.56 s windows (128 samples at 50 Hz) with 50% overlap. Assert the window length, and document why it matters: overlapping windows from one subject are near-duplicates, so a random window-level split would leak; the subject-wise split prevents this.
- `ChannelStandardizer`: per-channel mean and std over all windows and time steps of the **train-mask** data; std floored at 1e-8.
- `build_raw_loaders`: load, mask, fit the standardiser on train only, transform all three, build DataLoaders as in 5.2; `meta` includes the standardiser stats. Accept the `augment` argument now: if `True` before M3.4 exists, raise `NotImplementedError("augmentation lands in M3.4")`.

Tests: shapes `(7352, 128, 9)` and `(2947, 128, 9)`; labels 0..5; subjects match `subject_*.txt`; standardised train data has per-channel mean ≈ 0 and std ≈ 1; standardiser round-trips through `to_dict`/`from_dict`; loaders yield the right shapes. (The raw/feature alignment test lives in M1.1.)

Indicative commits: `feat(data): raw 9-channel inertial windows in fixed channel order`; `feat(data): per-channel standardiser fitted on training subjects`; `feat(data): raw-signal DataLoaders with subject-wise validation split`.

#### M3.4 Augmentation, off by default (branch `addyan/augment`)

Files: `src/har/data/augment.py`, update `build_raw_loaders`, `tests/test_augment.py`.

Transforms on standardised `(128, 9)` tensors, training set only: jitter (Gaussian noise, sigma 0.05) and per-channel scaling (factor from N(1, 0.1)), each applied with probability 0.5 when `data.augment: true`. Default `false`; all comparison runs keep it off so every model sees identical data. It is a possible Phase 3 ablation.

Tests: output shape unchanged; deterministic for a fixed generator; `augment=False` leaves data unchanged.

Indicative commits: `feat(data): jitter and scaling augmentation (disabled by default)`.

#### M3.5 Dataset report (branch `addyan/data-report`)

File: `scripts/data_report.py`, generating `docs/dataset_stats.md` (windows per class and split, subjects per split), `docs/figures/class_distribution.png` (windows per class for train, val and test), `docs/figures/sample_windows.png` (one example window per activity, body_acc x/y/z).

Indicative commits: `feat(scripts): dataset statistics and figures`; `docs(data): generated dataset statistics`.

#### M3.6 Model 3, BiLSTM with GRU ablation (branch `addyan/birnn`)

Files: `src/har/models/birnn.py`, `configs/bilstm.yaml`, `configs/gru.yaml`, `tests/test_birnn.py`, `report/interim/sections/model_bilstm.md`, then `results/bilstm/<run_id>/` and `results/gru/<run_id>/`.

Architecture: 7.3. Tests: shapes for both cells; parameter counts match 7.3; fits synthetic data; batch size 1; `pooling="mean"` and `bidirectional=False` variants work. Smoke runs, full runs for both configs, commit both run folders, write `model_bilstm.md` (covering the GRU ablation) from the actual results.

Indicative commits: `feat(models): bidirectional recurrent classifier (LSTM/GRU)`; `exp(bilstm): preliminary run on official split`; `exp(gru): GRU ablation preliminary run`; `docs(report): BiLSTM and GRU notes`.

#### M3.7 Contribution log generator (branch `addyan/contrib-log`)

File: `scripts/contribution_log.py`, output `docs/contribution_log.md`. Spec in 11.1.

Indicative commits: `feat(scripts): contribution log generated from git history`; later `docs: regenerate contribution log` (final-hour step 4).

#### M3.8 Literature review rows and synthesis (branch `addyan/lit`)

Files: `docs/lit/addyan.md` with the 4 papers allocated to M3 in 9.2; `report/interim/sections/lit_summary.md` (2 to 3 sentences tying all 13 rows together, written once M1's and M2's rows are merged; budget in 10.3). Commits: `docs(lit): rows for recurrent HAR models`; `docs(report): literature synthesis`.

#### M3.9 Raw preprocessing section and RNN risks (branch `addyan/report-preproc`)

Files: `report/interim/sections/preprocessing_raw.md` (10.3) and your bullets under `## Risks: Model 3 (M3)` in `risk_plan.md`. Commit: `docs(report): raw-signal preprocessing section and RNN risks`.

---

## 7. Model specifications

All four models are deliberately sized in the same range (roughly 100k to 200k trainable parameters) so differences come from the architecture family rather than raw capacity. The parameter counts below were computed from these exact definitions in PyTorch; each owner's test asserts them.

### 7.1 Model 1: MLP on engineered features (M1)

```
input (B, 561)
Linear(561, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.3)
Linear(256, 128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.3)
Linear(128, 6)   -> logits
```
Trainable parameters: **178,310**.

```yaml
# configs/mlp.yaml
name: mlp
owner: M1
data:
  representation: features
model:
  target: har.models.mlp.MLP
  params: {input_dim: 561, hidden: [256, 128], dropout: 0.3, n_classes: 6}
```
Role in the comparison: the feed-forward baseline. It sees hand-crafted time and frequency features, so it tests how far expert features go without any sequence modelling. Reference point: SVM on the same features, 96% test accuracy (Anguita et al., 2013).

### 7.2 Model 2: 1D-CNN on raw windows (M2)

```
input (B, 128, 9) -> permute -> (B, 9, 128)
Conv1d(9, 64, k=5, pad=2)    -> BatchNorm1d -> ReLU
Conv1d(64, 64, k=5, pad=2)   -> BatchNorm1d -> ReLU -> MaxPool1d(2)      # (B, 64, 64)
Conv1d(64, 128, k=5, pad=2)  -> BatchNorm1d -> ReLU
Conv1d(128, 128, k=5, pad=2) -> BatchNorm1d -> ReLU -> MaxPool1d(2)      # (B, 128, 32)
Conv1d(128, 128, k=3, pad=1) -> BatchNorm1d -> ReLU
AdaptiveAvgPool1d(1) -> flatten -> Dropout(0.5) -> Linear(128, 6)
```
Trainable parameters: **197,702**.

```yaml
# configs/cnn1d.yaml
name: cnn1d
owner: M2
data:
  representation: raw
model:
  target: har.models.cnn1d.CNN1D
  params: {in_channels: 9, channels: [64, 64, 128, 128, 128], kernel_sizes: [5, 5, 5, 5, 3],
           pool_after: [1, 3], dropout: 0.5, n_classes: 6}
```
(`pool_after` lists the zero-based conv indices followed by MaxPool1d(2); padding is `k // 2`.)

Role: tests whether features learned by local convolutions over the raw signal can replace the 561 hand-crafted ones. Reference point: 94.79% on raw UCI HAR (Ronao and Cho, 2016).

### 7.3 Model 3: Bidirectional LSTM, with GRU ablation (M3)

```
input (B, 128, 9)
LSTM(input_size=9, hidden_size=64, num_layers=2, bidirectional=True,
     batch_first=True, dropout=0.3)              # nn.GRU when cell == "gru"
pooling="last": concat(h_n[-2], h_n[-1]) -> (B, 128)   # final fwd and bwd states of the top layer
                (if bidirectional=False: h_n[-1] -> (B, 64))
pooling="mean": mean over time of the outputs -> (B, 128)
Dropout(0.3) -> Linear(128, 6)
```
For the LSTM, `h_n` is the first element of the returned `(h_n, c_n)` tuple. Trainable parameters: **138,502** (LSTM), **104,070** (GRU).

```yaml
# configs/bilstm.yaml
name: bilstm
owner: M3
data:
  representation: raw
model:
  target: har.models.birnn.BiRNN
  params: {in_channels: 9, hidden: 64, num_layers: 2, cell: lstm, bidirectional: true,
           dropout: 0.3, pooling: last, n_classes: 6}
```
```yaml
# configs/gru.yaml  (identical except name and cell)
name: gru
owner: M3
data:
  representation: raw
model:
  target: har.models.birnn.BiRNN
  params: {in_channels: 9, hidden: 64, num_layers: 2, cell: gru, bidirectional: true,
           dropout: 0.3, pooling: last, n_classes: 6}
```
Role: models order and duration across the window with gated recurrence; the GRU ablation checks whether the LSTM's extra gate matters on 2.56 s windows. Reference points: 93.6% with a residual BiLSTM (Zhao et al., 2018), 96.7% with a 4-layer LSTM (Murad and Pyun, 2017).

### 7.4 Model 4: Transformer encoder (M1)

```
input (B, 128, 9)
Linear(9, 64)                                   # input projection to d_model
+ learned positional embedding, shape (1, 128, 64), init N(0, 0.02)
Dropout(0.1)
TransformerEncoder(TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256,
                   dropout=0.1, batch_first=True, norm_first=True, activation="gelu"),
                   num_layers=3, enable_nested_tensor=False)
LayerNorm(64) -> mean over time -> Linear(64, 6)
```
Trainable parameters: **159,302**.

```yaml
# configs/transformer.yaml
name: transformer
owner: M1
data:
  representation: raw
model:
  target: har.models.transformer.TransformerHAR
  params: {in_channels: 9, seq_len: 128, d_model: 64, nhead: 4, num_layers: 3,
           dim_feedforward: 256, dropout: 0.1, n_classes: 6}
```
Pre-norm (`norm_first=True`) kept training stable at the shared lr of 1e-3 without a warm-up schedule in the dry run. `enable_nested_tensor=False` avoids a PyTorch warning with pre-norm layers. Role: replaces recurrence with self-attention over all 128 time steps. Reference point: 99.2% on KU-HAR (Dirgová Luptáková et al., 2022), which used a non-subject-wise split, so it is not directly comparable (see 9.2).

---

## 8. Training, evaluation and results protocol

### 8.1 Protocol (identical for every model)

| Item | Value |
|---|---|
| Train / val / test | 15 training subjects (5,276 windows) / 6 validation subjects (2,076) from `configs/split.json` / 9 official test subjects (2,947) |
| Loss | CrossEntropyLoss (classes are roughly balanced, so no class weights) |
| Optimiser | Adam, lr 1e-3, weight decay 0 |
| Batch size | 64 |
| Epochs | max 50; early stopping on validation macro-F1, patience 10, best weights restored |
| Gradient clipping | global norm 1.0 |
| Seed | 42 (`set_seed` before data loading and model construction) |
| Augmentation | off |
| Model and run selection | validation set only |
| Test set | evaluated once on the restored best checkpoint, reported, never used for any decision |

Interim runs are **untuned, single seed**, and must be labelled that way in the report. Phase 3 adds tuning on validation and 3 seeds (42, 43, 44).

### 8.2 Run folder (committed)

`results/<model>/<run_id>/`:
- `config.resolved.yaml`
- `metrics.json` (schema below)
- `history.csv` (one row per epoch)
- `curves.png`
- `confusion_val.png`, `confusion_test.png` (test files absent for `--no-test` runs)
- `classification_report_val.txt`, `classification_report_test.txt`
- `best.pt` stays local (gitignored)

`metrics.json` schema (values shown are type placeholders, not results):
```json
{
  "model": "cnn1d",
  "run_id": "20261003-091500_ab12cd3",
  "git_sha": "ab12cd3",
  "git_dirty": false,
  "run_by": "Ishan Abhijit Saraf",
  "status": "preliminary, untuned, single seed",
  "representation": "raw",
  "params_trainable": 0,
  "model_size_mb": 0.0,
  "best_epoch": 0,
  "epochs_run": 0,
  "train_time_s": 0.0,
  "mean_epoch_time_s": 0.0,
  "val": {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "confusion": []},
  "test": {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "confusion": []},
  "latency": {"cpu_bs1": {}, "cpu_bs256": {}, "device_bs1": {}},
  "hardware": {},
  "val_subjects": [],
  "protocol_override": false
}
```
For `--no-test` runs, `"test"` is `null`.

### 8.3 History row

`epoch, train_loss, train_acc, val_loss, val_acc, val_macro_f1, lr, epoch_time_s`

### 8.4 Sanity checks (on validation, never on test)

Debug with `--no-test` and judge runs by validation metrics.

Calibration from the dry run of these exact definitions (seed 42, untuned): every model reached a **validation macro-F1 of about 0.96**, and test accuracy then fell between roughly **88% and 95%**. The validation subjects are easier than the test subjects, so a validation-to-test gap of several points is normal and is itself worth a sentence in the report. Published results on UCI HAR mostly sit between about 93% and 97.6% (see 9.2).

- **Validation macro-F1 below about 0.85**: almost certainly a bug. Check the list below.
- **Validation macro-F1 above 0.995**: suspect leakage (validation subjects in training, scaler fitted on all data).
- **SITTING vs STANDING confusion** appears for every model (both are static with the phone at the waist). It is a finding for the error analysis, not a bug.

Common bugs:
1. Labels not shifted from 1..6 to 0..5 (CUDA "device-side assert" or index errors).
2. Channels stacked in directory-listing order instead of `CHANNELS` order, or differently for train and test.
3. Scaler fitted on train + val (or all data).
4. Missing `model.eval()` during validation (BatchNorm and dropout in training mode).
5. CNN given `(B, 128, 9)` without permuting to `(B, 9, 128)`.
6. LSTM created without `batch_first=True`.
7. sklearn metrics called without `labels=list(range(6))`.

### 8.5 Interim results table (what M2's script produces and the report uses)

Columns as in M2.6. Training time and latency are measured on each member's own machine in the interim; the report must say so. The Phase 3 cost comparison is re-measured on one machine with `profile_all.py`.

---

## 9. Literature review source material (verified)

Every row below was checked against the paper, its abstract or the publisher's page on 2 Oct 2026. Write the rows **in your own words**; the "relevance" text is a suggestion. Do not add papers or numbers that are not verified. Ignatov (2018) was deliberately left out because its UCI HAR number could not be confirmed from the paper itself.

### 9.1 Row format for `docs/lit/<member>.md`

```markdown
| Paper (Author, Year) | Method | Dataset | Key Result | Relevance to Project |
|---|---|---|---|---|
| Ronao & Cho, 2016 | ... | ... | ... | ... |

## References
- C. A. Ronao and S.-B. Cho, "...," *Expert Syst. Appl.*, vol. 59, pp. 235-244, 2016, doi: 10.1016/j.eswa.2016.04.032.
```

Keep each cell short (Method and Dataset under 12 words, Key Result under 20, Relevance under 25) so the 13-row table fits the page budget. Reference entries are an unnumbered list; the report builder numbers them globally ([1] to [13] in table order, M1 then M2 then M3) and the dataset citation becomes [14].

### 9.2 Allocation and verified content

#### M1 Mayank (feature-based and attention-based work): 4 papers

| Paper | Method | Dataset | Key result (verified) | Suggested relevance |
|---|---|---|---|---|
| Anguita et al., 2013 | One-vs-all multiclass SVM, Gaussian kernel, on 561 hand-crafted features | UCI HAR (introduced here) | 96% overall test accuracy on 2,947 test windows; lowest class recall SITTING 88% | Defines the dataset and the classical bar the MLP baseline must reach; its SITTING errors foreshadow our expected confusion |
| Reyes-Ortiz et al., 2016 | TAHAR: probabilistic SVM plus temporal filtering, handles postural transitions | SBHAR (extended UCI HAR with transitions), PAMAP2, REALDISP | SBHAR system error 3.22% (transitions learned) and 3.64% (treated as unknown); PAMAP2 error 5.67% | Same lab and sensor setup; shows feature-based pipelines stay strong and that transitions are a known gap of fixed windows |
| Mahmud et al., 2020 | Self-attention model with sensor-modality attention, no recurrence | PAMAP2, OPPORTUNITY, Skoda, USC-HAD | Window-wise macro-F1: PAMAP2 0.96, Skoda 0.97, OPPORTUNITY 0.67, USC-HAD 0.55 | Evidence that attention alone can model sensor sequences; motivates the Transformer as Model 4 |
| Dirgová Luptáková et al., 2022 | Transformer encoder on standardised accelerometer and gyroscope sequences | KU-HAR (90 participants, 18 classes), augmented to 83,129 samples | 99.2% average accuracy vs 89.67% for a random forest baseline | Current attention-based state of the art; its random 70:15:15 non-subject-wise split illustrates the leakage risk our subject-wise protocol avoids |

References:
- D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, "A public domain dataset for human activity recognition using smartphones," in *Proc. 21st Eur. Symp. Artif. Neural Netw., Comput. Intell. Mach. Learn. (ESANN)*, Bruges, Belgium, 2013, pp. 437-442.
- J.-L. Reyes-Ortiz, L. Oneto, A. Samà, X. Parra, and D. Anguita, "Transition-aware human activity recognition using smartphones," *Neurocomputing*, vol. 171, pp. 754-767, 2016, doi: 10.1016/j.neucom.2015.07.085.
- S. Mahmud, M. T. H. Tonmoy, K. K. Bhaumik, A. K. M. M. Rahman, M. A. Amin, M. Shoyaib, M. A. H. Khan, and A. A. Ali, "Human activity recognition from wearable sensor data using self-attention," in *Proc. 24th Eur. Conf. Artif. Intell. (ECAI)*, Frontiers in Artificial Intelligence and Applications, vol. 325, 2020, pp. 1332-1339, doi: 10.3233/FAIA200236.
- I. Dirgová Luptáková, M. Kubovčík, and J. Pospíchal, "Wearable sensor-based human activity recognition with transformer model," *Sensors*, vol. 22, no. 5, Art. no. 1911, 2022, doi: 10.3390/s22051911.

#### M2 Ishan (convolutional and hybrid work, survey): 5 papers

| Paper | Method | Dataset | Key result (verified) | Suggested relevance |
|---|---|---|---|---|
| Yang et al., 2015 | CNN with temporal convolution and pooling over raw multichannel signals | OPPORTUNITY, Hand Gesture | OPPORTUNITY accuracy 87.0%, 82.5%, 85.8% (subjects 1 to 3, no smoothing); about 5% above best baseline | Early evidence that convolution over raw channels learns useful features; basis for Model 2 |
| Jiang & Yin, 2015 | 2-layer CNN on an "activity image" (2D DFT of stacked signals); DCNN+ adds an SVM | UCI HAR, USC-HAD, SHO | UCI HAR test accuracy: DCNN 95.18%, DCNN+ 97.59% | Direct UCI HAR CNN result; shows how much the input representation matters for CNNs |
| Ronao & Cho, 2016 | Deep 1D convnet on raw accelerometer and gyroscope signals, optional FFT input | UCI HAR | 94.79% test accuracy on raw data; 95.75% with temporal FFT features added | Closest published setup to our 1D-CNN on raw UCI HAR windows; main reference value for Model 2 |
| Wang et al., 2019 | Survey of deep learning for sensor-based activity recognition | Many (lists UCI Smartphone: 30 subjects, 6 activities, 10,299 samples) | No single model wins everywhere; CNNs best on UCI Smartphone; RNNs suit short ordered activities; hybrids often beat single models | Frames the core question of our comparison and predicts family-dependent strengths |
| Xia et al., 2020 | Two LSTM layers followed by convolution, global average pooling, batch norm | UCI HAR, WISDM, OPPORTUNITY | 95.78% on UCI HAR, 95.85% on WISDM, 92.63% on OPPORTUNITY | Hybrid recurrent-convolutional result on our dataset; reference for combining Models 2 and 3 in future work |

Note for M2: do not copy numbers from the survey's tables (its Table 5 lists Ronao and Cho at 94.61%, which does not match their abstract). Cite primary papers for numbers.

References:
- J. B. Yang, M. N. Nguyen, P. P. San, X. L. Li, and S. Krishnaswamy, "Deep convolutional neural networks on multichannel time series for human activity recognition," in *Proc. 24th Int. Joint Conf. Artif. Intell. (IJCAI)*, Buenos Aires, Argentina, 2015, pp. 3995-4001.
- W. Jiang and Z. Yin, "Human activity recognition using wearable sensors by deep convolutional neural networks," in *Proc. 23rd ACM Int. Conf. Multimedia (MM '15)*, Brisbane, Australia, 2015, pp. 1307-1310, doi: 10.1145/2733373.2806333.
- C. A. Ronao and S.-B. Cho, "Human activity recognition with smartphone sensors using deep learning neural networks," *Expert Syst. Appl.*, vol. 59, pp. 235-244, 2016, doi: 10.1016/j.eswa.2016.04.032.
- J. Wang, Y. Chen, S. Hao, X. Peng, and L. Hu, "Deep learning for sensor-based activity recognition: A survey," *Pattern Recognit. Lett.*, vol. 119, pp. 3-11, 2019, doi: 10.1016/j.patrec.2018.02.010.
- K. Xia, J. Huang, and H. Wang, "LSTM-CNN architecture for human activity recognition," *IEEE Access*, vol. 8, pp. 56855-56866, 2020, doi: 10.1109/ACCESS.2020.2982225.

#### M3 Addyan (recurrent work): 4 papers

| Paper | Method | Dataset | Key result (verified) | Suggested relevance |
|---|---|---|---|---|
| Ordóñez & Roggen, 2016 | DeepConvLSTM: 4 conv layers (64 filters) + 2 LSTM layers (128 cells) | OPPORTUNITY, Skoda | Weighted F1: OPPORTUNITY gestures 0.915 (with Null), locomotion 0.930 (without Null); Skoda 0.958 | Reference deep architecture for wearable HAR; shows recurrence on top of convolution helps |
| Hammerla et al., 2016 | 4,000+ experiments comparing DNN (MLP), CNN, LSTM and bidirectional LSTM | OPPORTUNITY, PAMAP2, Daphnet Gait | b-LSTM best on OPPORTUNITY (mean F1 0.745); CNN best on PAMAP2 (mean F1 0.937) | Closest prior comparison of our exact families; finds the best family depends on the activity type |
| Murad & Pyun, 2017 | Deep LSTM networks: unidirectional, bidirectional and cascaded | UCI HAR, USC-HAD, OPPORTUNITY, Daphnet FOG, Skoda | 96.7% accuracy on UCI HAR with a 4-layer unidirectional LSTM | Strongest recurrent result on our dataset; upper reference for Model 3 |
| Zhao et al., 2018 | Residual bidirectional LSTM, 3 stacked layers | UCI HAR, OPPORTUNITY | UCI HAR 93.6% accuracy, 93.5% F1; OPPORTUNITY F1 90.5% | Bidirectional LSTM on UCI HAR; direct reference for our BiLSTM design |

Notes for M3: Murad and Pyun describe the UCI input as including a magnetometer, but the public dataset has only the 9 accelerometer and gyroscope channels, so do not repeat that claim. For Zhao et al., use the journal numbers above (the arXiv version differs slightly).

References:
- F. J. Ordóñez and D. Roggen, "Deep convolutional and LSTM recurrent neural networks for multimodal wearable activity recognition," *Sensors*, vol. 16, no. 1, Art. no. 115, 2016, doi: 10.3390/s16010115.
- N. Y. Hammerla, S. Halloran, and T. Plötz, "Deep, convolutional, and recurrent models for human activity recognition using wearables," in *Proc. 25th Int. Joint Conf. Artif. Intell. (IJCAI)*, New York, NY, USA, 2016, pp. 1533-1540.
- A. Murad and J.-Y. Pyun, "Deep recurrent neural networks for human activity recognition," *Sensors*, vol. 17, no. 11, Art. no. 2556, 2017, doi: 10.3390/s17112556.
- Y. Zhao, R. Yang, G. Chevalier, X. Xu, and Z. Zhang, "Deep residual Bidir-LSTM for human activity recognition using wearable sensors," *Math. Probl. Eng.*, vol. 2018, Art. no. 7316954, 2018, doi: 10.1155/2018/7316954.

#### Dataset citation (README and report)

J. Reyes-Ortiz, D. Anguita, A. Ghio, L. Oneto, and X. Parra, "Human Activity Recognition Using Smartphones," UCI Machine Learning Repository, 2013, doi: 10.24432/C54S4K.

#### Method references for Part C (not part of the interim table)

- S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Comput.*, vol. 9, no. 8, pp. 1735-1780, 1997.
- M. Schuster and K. K. Paliwal, "Bidirectional recurrent neural networks," *IEEE Trans. Signal Process.*, vol. 45, no. 11, pp. 2673-2681, 1997.
- K. Cho et al., "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in *Proc. EMNLP*, 2014, pp. 1724-1734.
- A. Vaswani et al., "Attention is all you need," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 5998-6008.

These 13 + 1 dataset + 4 method references already exceed the 12-reference minimum for Part C, with all 13 overlapping the interim table.

---

## 10. Interim report assembly (Part B)

### 10.1 Target

A Word document of 3 to 5 pages (excluding the cover page) in the official template's Part B format, plus a PDF, built by `report/build_interim_report.py` from files in the repo so every number traces back to `results/`.

Output: `report/build/ICT4442_Interim_Report_HAR.docx` and `.pdf` (gitignored; see 10.4).

### 10.2 Builder behaviour (`report/build_interim_report.py`, M1)

Inputs: the template docx, `report/interim/meta.yaml`, `configs/team.yaml`, `report/interim/members.local.yaml`, `docs/lit/*.md`, `report/interim/sections/*.md`, `results/summary.csv`, `docs/tasks/*.md`, `docs/contribution_log.md`, figures from `docs/figures/` and `results/figures/`.

`meta.yaml`:
```yaml
team_no: "FILL_ME"      # builder refuses to run while this placeholder remains
title: "Comparative Evaluation of Deep Learning Architectures for Human Activity Recognition from Smartphone Inertial Sensor Signals"
title_status: "Confirmed from Synopsis"
repo_url: auto            # from `git remote get-url origin`, converted to https
month_year: "October 2026"
models:
  - {key: mlp,         label: "MLP (engineered features)",        owner: M1}
  - {key: cnn1d,       label: "1D-CNN (raw signal)",              owner: M2}
  - {key: bilstm,      label: "BiLSTM (raw signal)",              owner: M3}
  - {key: gru,         label: "BiGRU ablation (raw signal)",      owner: M3}
  - {key: transformer, label: "Transformer encoder (raw signal)", owner: M1}
```
Member names come from `configs/team.yaml`; registration numbers from `members.local.yaml`.

Steps, using `python-docx` on a copy of the template:
1. **Cover page:** match paragraphs by prefix, not exact text (the template's placeholders are inconsistent): `startswith("<TITLE OF THE PROJECT")` → title; `startswith("Synopsis/Interim/Final report on")` → `Interim report on`; `startswith("<NAME OF THE STUDENT-")` (three paragraphs; the second has a stray `===`, the third ends `Reg no.3>`) → `Name - Reg. No.` in member order; `startswith("<")` containing `MONTH AND YEAR` → `month_year`. Write into the first run of each paragraph and clear the other runs to keep formatting.
2. **Remove Part A:** delete every body element (paragraphs and tables) from the paragraph starting `PART A` up to, not including, the paragraph starting `PART B`.
3. **Remove Part C:** delete every body element from the paragraph starting `PART C` to the end of the body, **keeping the final `w:sectPr`** element (page setup).
4. **Part B heading:** replace its text with `INTERIM REPORT`. Delete the instruction paragraphs by prefix: `startswith("Interim report should be")` and `startswith("Minimum 8")`. (The template writes the ranges "3 to 5" and "8 to 10" with en dash characters, not hyphens, so never match the full strings exactly.)
5. **Header fields:** after the bold labels starting "Team No. and Names", "Title of the Project", "GitHub Repository Link" add the values in a non-bold run.
6. **Tables:** find each table by its header-row text, not by index: literature (Paper / Method / Dataset / Key Result / Relevance), models (Model / Owner / Status / Preliminary Metric / Notes), contribution log (Member Name / Reg. No. / Task(s) Completed / Signature). Fill rows by deep-copying an existing empty row (keeps borders and fonts) and delete unused empty rows. In the literature table, append the global reference number to the paper cell, e.g. `Ronao & Cho, 2016 [7]`.
7. **Section text:** insert each section file's paragraphs under its heading per the map in 10.3 (plain paragraphs and simple bullets; strip markdown syntax). Add a table caption above each table ("TABLE I. LITERATURE REVIEW SUMMARY") and a figure caption below each figure ("Fig. 1. ...").
8. **Contribution evidence:** below the contribution table, insert the per-member summary table from `docs/contribution_log.md` (commits, lines added/removed excluding generated files, PRs merged) and a sentence pointing to the repo history.
9. **Figures (two only):** `docs/figures/class_distribution.png` in section 2 and the test confusion matrix of the model with the best validation macro-F1 in section 3, width about 3.2 in.
10. **References:** append a "References" heading with the 13 literature references numbered [1] to [13] in table order plus the dataset citation [14], at 8 pt. If the document runs over 5 pages, drop this list first (Part B does not require it; the table already names each paper).
11. Save the docx; if `soffice` (LibreOffice) is on PATH, convert with `soffice --headless --convert-to pdf --outdir report/build <docx>`; otherwise tell the human to export the PDF from Word. If the PDF exists, count pages with `pypdf` and warn if the count minus the cover page is outside 3 to 5.

Fallback if template surgery fails: generate the same content with `pandoc` using the template as `--reference-doc`, then paste the cover page by hand. Either way, a human opens the docx and checks it before submission.

### 10.3 Section map, sources and word budgets

| Template heading | Content | Source files (owner) | Budget |
|---|---|---|---|
| Header fields | Team no., names and reg. nos., title, repo link | `meta.yaml`, `team.yaml`, `members.local.yaml` (M1) | |
| 1. Literature Review | Table I (13 rows), then the synthesis | `docs/lit/*.md` (all); `lit_summary.md` (M3) | table ~1.5 pages; synthesis ≤ 80 words |
| 2. Dataset Acquisition & Preprocessing | The template's five items, input shapes, shared protocol, Fig. 1 | `preprocessing_raw.md` (M3), `preprocessing_features.md` (M1), `evaluation_protocol.md` (M2) | ≤ 180 + 80 + 100 words |
| 3. Models Implemented So Far | Table II, one-line model summaries, observations, Fig. 2 | `results/summary.csv` (M2), `model_*.md` (owners), `observations.md` (M2) | ≤ 40 words per model; observations ≤ 120 words |
| 4. Individual Contribution Log | Table III (template table) + per-member commit summary | `docs/tasks/*.md` (each), `docs/contribution_log.md` (M3) | ~0.4 page |
| 5. Risk / Plan for Remaining Work | Bullets + timeline table | `risk_plan.md` (M1, M2, M3) | ≤ 250 words + table |
| References | [1] to [14] | `docs/lit/*.md` | ~0.3 page, dropped first if over length |

What each section file must cover:

- **`preprocessing_raw.md` (M3):** cleaning (automated download, file and shape verification, no missing values, label/subject alignment checked); encoding (labels 1..6 to 0..5 for cross-entropy, no one-hot needed); normalisation (per-channel z-score fitted on the 15 training subjects); train/val/test split (official 21/9 subjects; 6 validation subjects listed by ID, seed 42; window counts; why a random window-level split would leak given 50% overlap); augmentation (jitter and scaling implemented, disabled for all comparison runs); input shape 128 x 9 and channel order.
- **`preprocessing_features.md` (M1):** the 561 features (time and frequency domain, provided by the dataset authors), duplicate names handled, StandardScaler fitted on training subjects only, same split as the raw pipeline (alignment verified by test).
- **`evaluation_protocol.md` (M2):** shared training protocol (8.1), metrics (accuracy, macro-F1, per-class, confusion matrix, parameters, training time, CPU latency), test-set discipline, and that interim timings come from different machines.
- **`model_<key>.md` (each owner):** front matter plus one short summary, written after the run:
  ```markdown
  ---
  model: cnn1d
  status: "Implemented, preliminary run"
  notes: "5 conv layers + global average pooling; main confusion SITTING/STANDING"
  ---
  One or two sentences (≤ 40 words) describing the architecture and why it is in the comparison.
  ```
  `status` and `notes` fill Table II; the builder takes the Preliminary Metric from `results/summary.csv` as `Test acc XX.X% / macro-F1 0.XXX (val macro-F1 0.XXX)`. Notes must describe the actual run.
- **`observations.md` (M2):** 3 to 5 sentences, only from actual results: which representation leads so far, whether SITTING/STANDING dominates errors, the validation-to-test gap, labelled preliminary.
- **`lit_summary.md` (M3):** 2 to 3 sentences: feature-based SVMs reach about 96% on UCI HAR; deep models on this dataset report about 93% to 97.6%; published comparisons differ in protocol (and some use non-subject-wise splits), which is the gap this project fills.
- **`risk_plan.md`:** section 12 is the starting point; each member updates their own heading with what the preliminary results actually showed.

### 10.4 Registration numbers, signatures and what gets committed

- Registration numbers stay out of committed files. M1 creates `report/interim/members.local.yaml` (gitignored) on the machine that builds the report, copying the three numbers from the synopsis, e.g. `{M1: "<reg no>", M2: "<reg no>", M3: "<reg no>"}`.
- `report/build/` is gitignored because the built report carries the registration numbers on its cover. If the repo is private and the team wants the submitted PDF in the repo, the human can explicitly remove that ignore line.
- Signatures: each member signs the printed or PDF report themselves. Never commit signature images.

---

## 11. Contribution log and task status files

### 11.1 `scripts/contribution_log.py` (M3)

Generates `docs/contribution_log.md` from `git log` on `main`:
- Header: generation time, the HEAD sha it describes, repo URL.
- Summary table per member: commits (non-merge), lines added, lines removed, first and last commit dates, PRs merged (via `gh pr list --state merged --limit 1000 --json author,number,title` if `gh` is available, else omitted).
- Line counts **exclude generated and bulk files**: `results/`, `docs/contribution_log.md`, `docs/dataset_stats.md`, `docs/figures/`, `docs/BUILD_SPEC.md`, `CLAUDE.md`, `report/template/`, `*.png`. Report those separately as "generated / documentation files changed".
- Per member: chronological list of their commits (date, short sha, subject, link `https://github.com/<owner>/<repo>/commit/<sha>`) and the top-level paths they changed.
- Member identification by author name against `configs/team.yaml`. If one person committed under two emails (for example before fixing their config), that person adds a `.mailmap` line joining **their own** identities. `.mailmap` must never reassign one member's commits to another.
- Merge commits listed separately (they show who merged which PR).
- Implementation: `git log --no-merges --numstat --date=iso-strict --pretty=format:...` via `subprocess`; no third-party dependencies.

### 11.2 `docs/tasks/<member>.md` (pre-filled by M1 in M1.0; afterwards only the owner edits their file)

```markdown
# Tasks: Ishan Abhijit Saraf (M2)

| ID | Task | Status | Branch |
|----|------|--------|--------|
| M2.1 | Metrics and plots | Done | ishan/metrics |
| M2.2 | Profiling | In progress | ishan/profiling |
| ... | ... | Todo | |
| Lit | 5 literature rows | Todo | ishan/lit |
```
Status values: `Todo`, `In progress`, `Done`, `Deferred to Phase 3`. Separate files per member avoid merge conflicts. The builder turns `Done` rows into the "Task(s) Completed" cell, and the printed contribution table is what all three sign.

---

## 12. Risk and plan for remaining work (draft content for `risk_plan.md`)

Update with real observations before submission. Remove any point the results contradict.

**Remaining work**
- Model 4 (Transformer, M1): full training under the shared protocol (if not done as a stretch run); if it underperforms, report and analyse it rather than drop it.
- Hyperparameter tuning for all models on the validation split only, with an equal budget per model (same number of configurations) so the comparison stays fair.
- Final runs with 3 seeds (42, 43, 44), reported as mean ± standard deviation.
- Same-machine cost comparison: parameter count, training time and CPU latency for all models measured on one machine (`profile_all.py`).
- Error analysis (all): per-class F1 and confusion matrices across models; test whether SITTING/STANDING and stair-direction errors are shared by all architectures (intrinsic to the sensor representation, Objective 3) or specific to some.

**Expected challenges and mitigations**
- *Transformer data hunger (M1):* 15 training subjects (5,276 windows) is small for self-attention. Mitigation: small model (d_model 64, 3 layers), pre-norm, dropout; optional convolutional patch stem as an ablation; a weak result is still reported.
- *Early stopping on a small validation set (all):* best epochs can come very early; check curves before tuning.
- *Validation-to-test gap (all):* the 6 validation subjects may be easier than the 9 test subjects; report both and avoid over-reading validation differences.
- *Static-posture confusion:* SITTING vs STANDING is hard for any model using a waist-mounted phone. Analyse rather than "fix"; compare against the MLP, whose 561 features include gravity-angle features.
- *Single-seed variance:* interim numbers are single runs; differences of about 1% are not meaningful until 3-seed results exist.
- *Hardware differences:* interim timings come from different laptops; final timings will be measured on one machine.
- *Fairness:* identical split, preprocessing, optimiser, early stopping and metrics for all models; any protocol override is recorded in the run config.
- *Test-set discipline:* the test split is used only to report selected checkpoints.

**Timeline to Final (from the synopsis, adjusted)**

| Dates | Milestone | Owner |
|---|---|---|
| 5 to 12 Oct | Model 4 trained; tuning of all four models on validation | M1 (Model 4); each owner tunes their model |
| 13 to 19 Oct | Final 3-seed runs, comparison table, same-machine profiling | Each owner runs own model; M2 table and profiling |
| 20 to 25 Oct | Error analysis, Part C report draft (one methodology subsection per owner) | All |
| 26 to 29 Oct | Presentation and viva | All |
| 31 Oct | Final report (Part C) and repository submission | All |

---

## 13. Definition of done (interim)

Tick all before submitting (the final-hour sequence in 2.1 covers the last few):

- [ ] Repo on GitHub; Ishan and Addyan are collaborators; only merge commits allowed; CI green on `main`.
- [ ] Each member's commits appear under their own GitHub profile (avatar shown, not grey).
- [ ] `src/har/data/` is tracked (not swallowed by `.gitignore`).
- [ ] `python -m har.data.download` works on a fresh clone; `pytest -q` passes.
- [ ] `configs/split.json` committed with 6 validation subjects.
- [ ] Run folders committed for `mlp` (by M1), `cnn1d` (by M2), `bilstm` and `gru` (by M3); `transformer` if the stretch run happened.
- [ ] `results/summary.md` regenerated after the last run; README results block updated.
- [ ] `docs/lit/{mayank,ishan,addyan}.md` committed by their owners (13 rows) and `lit_summary.md` written.
- [ ] `docs/dataset_stats.md` and figures committed.
- [ ] All section files in 10.3 exist; `risk_plan.md` has bullets from all three members.
- [ ] `docs/tasks/*.md` statuses up to date.
- [ ] `docs/contribution_log.md` regenerated after the last code/results commit.
- [ ] Report built, opened in Word, 3 to 5 pages excluding cover, numbers match `results/summary.csv`, team number filled, signed by all three.
- [ ] Tag `interim-v1` pushed.

---

## 14. Troubleshooting

| Problem | Fix |
|---|---|
| Commits show a grey avatar on GitHub | The commit email is not on your GitHub account. Add that email to your account (GitHub, Settings, Emails); GitHub then links the already-pushed commits to you. Also fix `git config user.email` for future commits, and add a `.mailmap` line joining your own two identities for the log script. Do not rewrite history. |
| `import har.data` fails after cloning, or `src/har/data/` is missing on GitHub | `.gitignore` has `data/` instead of `/data/`. Fix the pattern, then `git add src/har/data` and commit. |
| `torch.cuda.is_available()` is False on the GPU laptop | Reinstall PyTorch with the CUDA command from pytorch.org matching your driver (`nvidia-smi` shows the supported CUDA version); uninstall the CPU build first. |
| DataLoader hangs on Windows | Keep `num_workers=0`; scripts must have `if __name__ == "__main__":`. |
| Download blocked or slow | Download the zip in a browser from the URL in M3.1 and extract the inner `UCI HAR Dataset` folder into `data/raw/`; then rerun `python -m har.data.download` (it verifies the existing folder and skips the download). |
| Path errors mentioning `UCI HAR Dataset` | The folder name has spaces; always use `pathlib.Path`. |
| PR shows merge conflicts | On your branch: `git fetch && git merge origin/main`, resolve keeping both sides' content, commit, push. Never force-push. |
| ProtocolError when training | Your model config contains `train` or `seed`. Remove it; the protocol lives in `configs/base.yaml`. Phase 3 tuning uses `--allow-protocol-override`, which is recorded. |
| BiLSTM or Transformer slow on CPU | Expected (sequential steps / attention over 128 steps). Run once at full settings; or train on Colab with your own git name set and commit the downloaded run folder from your own machine. |
| Validation metrics suspiciously low or high | Section 8.4 checklist. |
