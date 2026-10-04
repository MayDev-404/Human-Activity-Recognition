# CLAUDE.md: HAR Deep Learning Comparison (ICT 4442 Mini Project)

**Project:** Comparative Evaluation of Deep Learning Architectures for Human Activity Recognition from Smartphone Inertial Sensor Signals.

**Team**

| ID | Member | Owns |
|----|--------|------|
| M1 | Mayank Kejariwal | Model 1 (MLP), Model 4 (Transformer), engineered-feature pipeline, repo scaffold, report assembly |
| M2 | Ishan Abhijit Saraf | Model 2 (1D-CNN), shared evaluation harness (metrics, trainer, train script, timing, parameter counts, results table) |
| M3 | Addyan Kumar | Model 3 (BiLSTM + GRU ablation), dataset download, raw-signal loading, normalisation, validation split, augmentation, contribution log |

The full specification is in `docs/BUILD_SPEC.md`. Read it completely before starting any task. This file holds the rules that apply to every session.

---

## 1. Confirm who you are working for (every session, before any code)

1. Run `git config user.name`, `git config user.email` and `gh auth status`, and show the results to the human.
2. Ask which member they are (M1, M2 or M3) unless they already said so.
3. If the name does not match that member, the email is not one linked to their own GitHub account, or `gh` is logged into someone else's account, stop and help them fix **their own** setup (on their own machine: `git config --global user.name "..."`, `git config --global user.email "..."`, `gh auth login`; in the shared workspace: see below). Do not continue until they confirm.
4. Work only on that member's tasks from `docs/BUILD_SPEC.md` section 6, following the **"Order"** line at the top of that member's section (it takes precedence over numeric order).

### Shared workspace (all three members use this folder on Mayank's laptop)

- One member at a time. Before starting, `git status` must be clean. If it shows another member's uncommitted changes, stop and ask the human; never commit, stash or discard someone else's work.
- Logging in to `gh` does **not** change who commits are credited to: git takes the author from `user.name` and `user.email`, and the global values on this laptop are Mayank's. So at the start of every session, set the identity of the member at the keyboard **for this repository only** (never `--global`, which would change Mayank's identity everywhere): `git config user.name "<full name>"` and `git config user.email "<email verified on that member's GitHub account>"`.
- `git push` and `gh pr create` act as the **active** `gh` account (git uses `gh` as its credential helper here). Add each account once with `gh auth login`, then `gh auth switch --user <username>` at the start of each session.
- Re-run the three checks in step 1 and show them to the human before the first commit of the session.
- Environment on this laptop: use the existing `.venv` (Python 3.12). Install only the **CPU** PyTorch build; Windows Smart App Control blocks the CUDA build's DLLs.

## 2. Git rules (contribution is graded from the commit history)

- Commit only as the configured local user. Never pass `--author`, never set `GIT_AUTHOR_*` or `GIT_COMMITTER_*` variables, never pass `--date`, never change commit dates.
- Never rewrite pushed history: no force-push, and no amend, rebase or squash of commits that are already pushed. To bring a pushed branch up to date, use `git merge main`. Rebasing your own unpushed local commits onto a fresh `main` is fine.
- Merge PRs with **"Create a merge commit"**, never "Squash and merge", so every individual commit stays in `main`.
- Tasks are chained, so do not wait on reviews: once `pytest -q` passes locally and CI is green, the author may merge their own PR immediately. Teammates review afterwards when they can (review comments and approvals also count as contribution evidence).
- Never commit on behalf of another member. Do not edit files owned by another member (ownership column in BUILD_SPEC section 3). If you need a change in someone else's file, tell the human so they can ask the owner. **Exception:** if a bug in someone else's file blocks you and the owner cannot be reached, you may open a small `fix:` PR for that file; the owner reviews it afterwards.
- Commit as you actually work, one logical change per commit (a function or module with its tests, a config, a results run, a doc section). The commit lists in BUILD_SPEC are indicative; never split finished work into pieces to match them, and never create empty or cosmetic commits to inflate counts. Prefixes: `feat`, `fix`, `test`, `docs`, `exp`, `chore`, `refactor`, `ci`.
- Branch per task, named `<member>/<topic>` (for example `ishan/metrics`). Before starting a task: `git switch main && git pull`. When done: `pytest -q`, push, `gh pr create --fill` (or with `--title` and `--body`), and tell the human.
- Keep Claude Code's default commit attribution trailer. AI assistance is disclosed in the README.
- Never commit `/data/`, `*.pt` checkpoints, `.venv/`, `results/_smoke/`, `*.local.yaml` files, `report/build/` (unless the repo is private and the human asks), or secrets.

## 3. Honesty rules for results and citations

- Every number that appears in docs, README or the report must come from a file under `results/` produced by an actual run. Never type, estimate, round up or invent a metric.
- Commit every full (non-smoke) run you make. The results table selects each model's run by best **validation** macro-F1, never by test score.
- Never invent references. Use only the verified papers in BUILD_SPEC section 9, or a paper the human supplies whose DOI or URL you have opened and checked.
- The test split is only for reporting the selected checkpoint. While debugging, train with `--no-test` and judge runs on validation metrics. Never pick hyperparameters, epochs or architectures by looking at test results.
- If a model underperforms, report it as it is. A weak result with a good explanation is a valid finding.

## 4. Fixed protocol (changes need agreement from all three members)

- **Dataset:** UCI HAR, "Human Activity Recognition Using Smartphones" (UCI id 240, DOI 10.24432/C54S4K). Official subject-independent split: train = 21 subjects (7,352 windows), test = 9 subjects (2,947 windows).
- **Validation:** 6 of the 21 training subjects, chosen once with seed 42 and stored in `configs/split.json` (committed). Everyone reads that file; nobody regenerates it with a different seed.
- **Inputs:** `features` representation = (N, 561) engineered features (MLP). `raw` representation = (N, 128, 9) windows (CNN, RNN, Transformer). Channel order is fixed: `body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z, total_acc_x, total_acc_y, total_acc_z` (defined once in `har/data/constants.py`).
- **Labels:** file values 1..6 map to 0..5 = `WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING`.
- **Normalisation:** statistics are fitted on the 15 remaining training subjects only, then applied to validation and test.
- **Training:** Adam, lr 1e-3, batch size 64, CrossEntropyLoss, max 50 epochs, early stopping on validation macro-F1 (patience 10, best weights restored), gradient clipping at norm 1.0, seed 42. These values live only in `configs/base.yaml`. Model configs must not contain `train` or `seed` keys (the config loader enforces this).
- **Metrics:** accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, trainable parameters, training time, CPU inference latency at batch size 1 with 1 thread. Always pass `labels=list(range(6))` (and `zero_division=0`) to sklearn metric functions.
- **Augmentation:** implemented but **off** for all comparison runs.

## 5. Environment and commands

- Python 3.10 to 3.13, in a venv.
- **Install PyTorch first.** NVIDIA GPU: use the CUDA command from https://pytorch.org/get-started/locally/ for your CUDA version, then check `python -c "import torch; print(torch.cuda.is_available())"`. CPU only: `pip install torch`. Then `pip install -r requirements.txt && pip install -e .`
- Download data: `python -m har.data.download`
- Tests: `pytest -q` (tests that need the dataset skip automatically when it is missing)
- Train: `python scripts/train.py --config configs/<model>.yaml` (add `--no-test` while debugging)
- Smoke test (writes to `results/_smoke/`, never committed): `python scripts/train.py --config configs/<model>.yaml --smoke`
- Results table: `python scripts/make_results_table.py`
- Device is chosen automatically (cuda, then mps, then cpu). DataLoaders use `num_workers=0` so they work on Windows.

## 6. Code and writing conventions

- Package code lives in `src/har/`. Scripts in `scripts/` only parse arguments and call package functions, and have an `if __name__ == "__main__":` guard.
- Type hints and short docstrings on every public function. Shapes are documented in docstrings, for example `(B, 128, 9) -> (B, 6)`.
- Every module ships with tests in `tests/`, written by the module's owner. Tests of models that contain BatchNorm run batch-size-1 checks in `eval()` mode.
- Notebooks are allowed in `notebooks/` for exploration only. Nothing imports from them and no reported number comes from them.
- Paths come from `har.utils.paths`; never hard-code absolute paths. The dataset folder name contains spaces (`UCI HAR Dataset`), so always use `pathlib.Path`.
- Do not use em dashes (the long dash, Unicode U+2014) in README, docs or report text. Use commas, colons or parentheses instead.

## 7. Shared laptop
The team shares one laptop. At the start of each session the person at the keyboard
sets the repo-local git identity (`git config user.name` / `user.email`, never --global)
to their own, and Claude confirms it with them before any commit. Pushing and PRs may use
whichever gh account is logged in; commit authorship comes from the commit email. Never
commit while the identity set belongs to someone other than the person doing the work.
