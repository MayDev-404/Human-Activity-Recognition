# PHASE3_SPEC.md: Final Phase (Part C report, final results, repo submission)

This spec extends `docs/BUILD_SPEC.md` and `CLAUDE.md` for Phase 3. Everything in those files still applies unless a decision below changes it. Read all three before any Phase 3 task.

**Status at the start of Phase 3 (from the 23 Oct audit):** Phase 2 complete and tagged `interim-v1`; 241 tests pass; one untuned seed-42 run per model; no Phase 3 work yet.
**Deadlines:** Part C due **Sat 31 Oct 2026**. Presentation in the 26 to 29 Oct window (exact slot: ask the team).
**Hardware:** all training on Mayank's laptop (RTX 4060 Laptop GPU, Ryzen 7 7840HS). The team shares this laptop; see section 7 for handovers.

| ID | Member | GitHub | Phase 3 ownership |
|----|--------|--------|-------------------|
| M1 | Mayank Kejariwal | MayDev-404 | MLP + Transformer tuning and final runs; references; Part C builder; Abstract, Keywords, I, III, IV-A, IV-D, VII, Acknowledgement; final build and tag |
| M2 | Ishan Abhijit Saraf | IshanCodes21 | Harness changes (seeds, tuning runner, predictions, final aggregation); CNN tuning and final runs; same-machine profiling; results figures; IV-B, V, VI-A |
| M3 | Addyan Kumar | Addhyanitis | BiLSTM + GRU tuning and final runs; error analysis; GRU latency check; II, IV-C, VI-B, VIII; final contribution log |

---

## 1. Decisions (agreed by all three)

These change the Phase 2 protocol, so they need all three members. Agreement is recorded by **both other members approving** the PR that implements them (F2.1).

- **D1 Seeds.** Final runs use seeds **42, 43 and 44**, set only with `scripts/train.py --seed N`. The allowed list lives in `configs/base.yaml` as `final_seeds: [42, 43, 44]`. A seed from that list is **not** a protocol override. Any other seed still requires `--allow-protocol-override`. The seed is recorded in `metrics.json` as `"seed"`.
- **D2 Tuning scope and budget.** The shared training protocol (Adam, lr 1e-3, batch 64, max 50 epochs, early stopping on val macro-F1 with patience 10, clip 1.0) stays **identical for every model**. Tuning changes only `model.params`, using exactly the grids in section 3: **4 configurations per model** (including the Phase 2 default), seed 42, validation only (`--no-test`), no latency profiling. Selection: best validation macro-F1; ties (within 0.001) go to fewer parameters, then to the default. The winner is written to `configs/final/<model>.yaml`.
- **D3 Final runs.** `configs/final/<model>.yaml` × seeds 42, 43, 44, tag `final`, with test evaluation and profiling. Results are reported as **mean ± sample standard deviation (n = 3)**. The test set is still only used for reporting.
- **D4 Cost comparison.** Parameter counts come from the model. Training time is mean ± std over the 3 final runs. CPU latency comes from **one** `profile_all.py` run in a single process on the final configs (label `rtx4060-laptop`), replacing the per-run latencies that were measured in separate processes.
- **D5 GRU** stays a separate ablation row next to the BiLSTM.
- **D6 Augmentation ablation** is out of scope for Phase 3 (time). It goes into Future Work.
- **D7 Predictions.** Every non-smoke run saves its per-window predictions so the error analysis can compare models window by window.
- **D8 Submission.** Final tag `final-v1`. `report/build/` stays gitignored (the repo is public and the built report carries registration numbers). Author emails for the C.4 author block go in the gitignored `report/interim/members.local.yaml`.
- **D9 Reviews.** Every Phase 3 PR is approved by a different member, using that member's own `gh` account, before it is merged (`gh pr review <n> --approve`, then `gh pr merge <n> --merge`). The reviewer reads the diff summary before approving.

---

## 2. Tasks per member

Task IDs start with F (final). Each member adds their own F rows to their `docs/tasks/<member>.md` in their first Phase 3 PR and updates them as tasks finish. Commit plans are indicative, as in BUILD_SPEC.

### M1 Mayank

**F1.1 Phase 3 setup and hygiene** (branch `mayank/phase3-setup`)
- Commit this file as `docs/PHASE3_SPEC.md`.
- Add `final_seeds: [42, 43, 44]` to `configs/base.yaml` (D1).
- Append the CLAUDE.md amendment in section 8 of this file.
- Delete the untracked duplicate `ICT_4442_Mini_Project_Report_Template.docx` at the repo root (never commit it).
- README: remove the stale line saying scripts "are added by later tasks"; fix the clone URL to `MayDev-404/Human-Activity-Recognition` in README and BUILD_SPEC 3 and 4.4.
- Interim builder guard: `load_meta` must reject empty values, `FILL_ME`, `TBD` and any `<...>` placeholder in non-draft builds, with a test. Set `team_no` in `report/interim/meta.yaml` to the confirmed team number.
- Add F1.x rows to `docs/tasks/mayank.md`.

**F1.2 References** (branch `mayank/references`)
- Create `report/final/references.yaml`: one entry per key in section 6.3, value = the IEEE string from BUILD_SPEC 9.2 (copy exactly; these were verified).
- Add `report/final/README.md` explaining citation syntax `[@key]` and number placeholders (section 6.4).

**F1.3 Tuning and final runs: MLP and Transformer** (branch `mayank/final-runs`; needs F2.1 merged)
- `python scripts/tune.py --model mlp` and `--model transformer` (section 4.2). Commit `results/tuning/<model>/` and the generated `configs/final/<model>.yaml`.
- Final runs: `python scripts/train.py --config configs/final/<model>.yaml --seed <s> --tag final` for s in 42, 43, 44. Commit the six run folders.

**F1.4 Part C builder** (branch `mayank/final-builder`)
- `report/build_final_report.py` and `tests/test_final_report_builder.py`, per section 6.6. Supports `--draft` from the start so it can be built before all sections exist.

**F1.5 M1 sections and own contribution statement** (branch `mayank/final-sections`)
- `abstract.md`, `keywords.md`, `1_introduction.md`, `3_objectives.md`, `4a_mlp.md`, `4d_transformer.md`, `7_conclusion.md`, `acknowledgement.md` (section 6.2).
- `report/final/contributions/mayank.md`, written by Mayank personally (section 6.5).

**F1.6 Final build, tag, submit** (final-hour sequence, section 9).

### M2 Ishan

**F2.1 Harness changes for Phase 3** (branch `ishan/phase3-harness`; both M1 and M3 approve, D1)
Section 4.1. Also add F2.x rows to `docs/tasks/ishan.md`.

**F2.2 Tuning and final runs: 1D-CNN** (branch `ishan/final-runs`)
Same commands as F1.3 for `cnn1d`.

**F2.3 Final table, profiling, results figures** (branch `ishan/final-table`; after all 15 final runs exist)
- `python scripts/make_results_table.py` (writes the final table, section 4.1 item 6).
- `python scripts/profile_all.py --label rtx4060-laptop` on the final configs (D4). Commit `results/profile_rtx4060-laptop.csv`.
- `python scripts/plot_curves.py` plus the two new figures in section 4.3.
- Commit everything; the README results block now shows the final table.

**F2.4 M2 sections and own contribution statement** (branch `ishan/final-sections`)
- `4b_cnn1d.md`, `5_setup.md`, `6a_results.md` (section 6.2).
- `report/final/contributions/ishan.md`, written by Ishan personally.

### M3 Addyan

**F3.1 Tuning and final runs: BiLSTM and GRU** (branch `addyan/final-runs`; needs F2.1 merged)
Same commands as F1.3 for `bilstm` and `gru`. Also add F3.x rows to `docs/tasks/addyan.md`.

**F3.2 Error analysis** (branch `addyan/error-analysis`; after all 15 final runs exist)
`src/har/eval/errors.py`, `scripts/error_analysis.py`, `tests/test_errors.py`, outputs in `results/analysis/` (section 5).

**F3.3 GRU latency check** (same branch or `addyan/gru-latency`)
Using the single-process profile CSV from F2.3: if the GRU is still much slower than the BiLSTM at batch 1 on CPU, measure both inside and outside `with torch.backends.mkldnn.flags(enabled=False):`. Record the outcome in `results/analysis/gru_latency_check.md` (numbers from this check may be quoted only from that file). Remove or correct the unsupported claim in `report/interim/sections/risk_plan.md` if the check does not support it.

**F3.4 M3 sections and own contribution statement** (branch `addyan/final-sections`)
- `2_literature.md`, `4c_birnn.md`, `6b_error_analysis.md`, `8_future_work.md` (section 6.2).
- `report/final/contributions/addyan.md`, written by Addyan personally.

**F3.5 Final contribution log** (final-hour sequence, section 9): regenerate `docs/contribution_log.md` as the last commit before the final build.

---

## 3. Tuning grids (equal budget: 4 configurations per model)

Each grid is the full product of two parameters; all other `model.params` stay at the Phase 2 values. The first configuration is the Phase 2 default. Parameter counts were computed from these definitions; all stay within roughly 100k to 215k, so the comparison remains capacity-matched.

| Model | Parameter A | Parameter B | Params per config |
|---|---|---|---|
| mlp | `dropout` ∈ {0.3, 0.5} | `hidden` ∈ {[256, 128], [256, 256]} | 178,310 / 212,230 |
| cnn1d | `dropout` ∈ {0.5, 0.3} | `kernel_sizes` ∈ {[5,5,5,5,3], [7,7,5,5,3]} | 197,702 / 207,046 |
| bilstm | `dropout` ∈ {0.3, 0.5} | `pooling` ∈ {last, mean} | 138,502 (all) |
| gru | `dropout` ∈ {0.3, 0.5} | `pooling` ∈ {last, mean} | 104,070 (all) |
| transformer | `dropout` ∈ {0.1, 0.2} | `dim_feedforward` ∈ {256, 128} | 159,302 / 109,766 |

Grid files: `configs/tuning/<model>.yaml`, for example
```yaml
model: mlp
base_config: configs/mlp.yaml
grid:
  dropout: [0.3, 0.5]
  hidden: [[256, 128], [256, 256]]
```
Grid files for all five models are created by M2 in F2.1 (they are part of the harness), exactly as in this table.

---

## 4. Harness changes (M2, F2.1) and results outputs

### 4.1 Changes

1. **`--seed` in `scripts/train.py`** (D1). Validated against `final_seeds` in base.yaml; an unlisted seed raises unless `--allow-protocol-override`. The runner applies it before data loading and model construction and writes `"seed"` into `metrics.json`. `protocol_override` stays false for listed seeds.
2. **`--no-profile`** skips latency profiling (used by tuning runs). `latency` is `null` in that case.
3. **Status label from the tag.** `tag == "final"` → `"final, tuned on validation, seed <s>"`; tag starting with `tune` → `"tuning run, validation only"`; otherwise the existing preliminary label. Interim runs are untouched.
4. **Predictions (D7).** Every non-smoke run writes `predictions_val.csv` and `predictions_test.csv` (columns `index, y_true, y_pred`; row order = loader order, which is not shuffled) into its run folder. Test file absent for `--no-test`.
5. **`scripts/tune.py --model <name>`** with logic in `src/har/train/tuning.py`: reads `configs/tuning/<name>.yaml`, expands the grid, runs each configuration with seed 42, `--no-test`, `--no-profile`, tag `tune-c<i>`, output root `results/tuning/`. Writes `results/tuning/<name>/tuning_summary.csv` (config id, params, val acc, val macro-F1, best epoch, run_id) and the selected configuration to `configs/final/<name>.yaml` (same `name`, `owner`, `data` as the base config; only `model.params` differ), applying the D2 tie rule. Refuses if `configs/final/<name>.yaml` already exists unless `--force`.
6. **Final aggregation in `src/har/eval/results.py`.**
   - The **interim** summary keeps working but only considers untagged runs, so `results/summary.md` stays exactly as submitted in Phase 2.
   - New **final** summary: for each model, the runs tagged `final`; require exactly seeds {42, 43, 44} (fail loudly otherwise). Write `results/final_summary.csv` (raw floats: mean and std columns) and `results/final_summary.md` with columns: Model, Run by, Representation, Params, Val Macro-F1, Test Acc, Test Macro-F1, Train time (s), CPU latency bs=1 (ms), Seeds. Values formatted `0.941 ± 0.004`; accuracy as `%`. CPU latency comes from `results/profile_rtx4060-laptop.csv` when present (mark "(single process)"), otherwise the mean of the runs' own latencies with a note.
   - The README block switches to the final table once it exists, with a one-line caption: "Final results: mean ± sample std over seeds 42, 43, 44 of the validation-selected configuration; shared protocol."
   - `profile_configs` / `profile_all.py` profile `configs/final/*.yaml` when they exist (fall back to `configs/*.yaml`).
7. **Tests** for every change above (seed validation, no-profile, status labels, predictions files, tuning expansion and tie rule on synthetic data, final aggregation including the missing-seed failure). All existing tests keep passing.

### 4.2 Commands each owner runs

```
python scripts/tune.py --model <name>
python scripts/train.py --config configs/final/<name>.yaml --seed 42 --tag final
python scripts/train.py --config configs/final/<name>.yaml --seed 43 --tag final
python scripts/train.py --config configs/final/<name>.yaml --seed 44 --tag final
```
Before trusting a run, apply the validation sanity checks in BUILD_SPEC 8.4. Commit every full run (tuning and final).

### 4.3 New results figures (M2, F2.3)

- `results/figures/final_val_curves.png`: validation macro-F1 per epoch for each model's seed-42 final run.
- `results/figures/cost_tradeoff.png`: test macro-F1 (mean ± std error bars) against single-process CPU latency at batch 1 (log x-axis), one labelled point per model, marker area proportional to parameter count.
- `results/figures/confusion_test_<model>_final.png`: test confusion matrix summed over the 3 seeds, row-normalised, for each model.

---

## 5. Error analysis (M3, F3.2)

Inputs: the 15 final run folders (`metrics.json` confusion matrices and `predictions_test.csv`). Outputs in `results/analysis/`, all generated by `scripts/error_analysis.py` (logic in `src/har/eval/errors.py`):

1. `per_class_f1.csv` / `.md`: test F1 per class per model, mean ± std over seeds. Figure `per_class_f1.png`: grouped bars with error bars.
2. `error_structure.csv` / `.md` per model (mean over seeds): total test errors; SITTING↔STANDING errors and their share of all errors; errors among the three walking classes; dynamic↔static errors; LAYING misclassifications and their main target class.
3. **Error overlap** (Objective 3, shared vs architecture-specific errors): for each seed, the set of misclassified test windows per model.
   - Pairwise Jaccard overlap of those sets between models, averaged over seeds → `error_overlap.csv` and heatmap `error_overlap.png`.
   - Number of windows misclassified by **all** five models, by exactly one model, and by none (mean over seeds), plus the class breakdown of the "all models wrong" windows.
4. **Shared vs model-specific confusion cells:** a cell (true → predicted) is *shared* if it holds at least 2% of that class's test windows for every model (summed over seeds), and *model-specific* if it does so for one or two models only. List both in `error_summary.json` and `.md`.
5. `error_summary.json`: every number above under stable keys (used by `{{analysis:...}}` placeholders, section 6.4).

These outputs answer Objective 3 directly: if most errors are shared across all families (high overlap, many "all wrong" windows, mostly SITTING/STANDING), the error is intrinsic to the sensor representation; if overlaps are low or confusions are family-specific (for example LAYING→UPSTAIRS only in the recurrent models), it is architecture-dependent. Write whichever the data shows.

---

## 6. Part C report

### 6.1 Structure (template Part C, in order)

Cover page (template, "Final report on") → title block (title, authors, affiliation, emails) → Abstract → Keywords → I. Introduction → II. Literature Review → III. Objectives → IV. Methodology (IV-A MLP, IV-B 1D-CNN, IV-C BiLSTM with GRU ablation, IV-D Transformer) → V. Experimental Setup → VI. Results and Discussion (VI-A comparison and why performance differs, VI-B error analysis) → VII. Conclusion → VIII. Future Work → Acknowledgement (unnumbered) → References → C.3 Individual Contribution Statement (separate final page).

### 6.2 Section files, owners, budgets, content

All in `report/final/sections/`. Write only from repo facts: numbers through placeholders (6.4), papers through citation keys (6.3). Keep each within budget.

| File | Owner | Budget (words) | Must contain |
|---|---|---|---|
| `abstract.md` | M1 | 150 to 250 | Problem, four families + GRU ablation, dataset and subject-independent protocol, key results (placeholders), main finding |
| `keywords.md` | M1 | 5 to 8 terms | Comma-separated, e.g. human activity recognition, inertial sensors, deep learning, convolutional neural networks, LSTM, Transformer, model comparison |
| `1_introduction.md` | M1 | 300 to 400 | HAR context and applications, why published numbers are hard to compare (protocol differences, random splits leak subjects), scope of this project, contribution summary |
| `2_literature.md` | M3 | 450 to 550 | The 13 interim papers turned into grouped prose (feature-based, CNN, recurrent, attention, comparisons/surveys), **at least 10 cited**, ending with the gap this project fills |
| `3_objectives.md` | M1 | 80 to 140 | The 3 synopsis objectives, crisp and measurable (performance under one protocol; accuracy vs cost; shared vs architecture-specific errors) |
| `4a_mlp.md` | M1 | 200 to 280 | Architecture, final hyperparameters (from `configs/final/mlp.yaml` and the tuning summary), training configuration, why it is in the comparison |
| `4b_cnn1d.md` | M2 | 200 to 280 | Same for the 1D-CNN |
| `4c_birnn.md` | M3 | 250 to 330 | Same for the BiLSTM, plus the GRU ablation and why it is included |
| `4d_transformer.md` | M1 | 200 to 280 | Same for the Transformer |
| `5_setup.md` | M2 | 350 to 450 | Dataset, preprocessing (both representations), split with subject IDs and window counts, shared protocol, tuning procedure and equal budget (Table II), seeds, metrics, test-set discipline, hardware |
| `6a_results.md` | M2 | 400 to 500 | Table I discussion; **why performance differs** (hand-crafted features vs learned; local convolution vs recurrence vs attention; data size; val-to-test gap; seed variance); accuracy-vs-cost trade-off (Fig. cost_tradeoff) for on-device use (Objective 2) |
| `6b_error_analysis.md` | M3 | 400 to 500 | Per-class results (Table III), error structure, error overlap and the "all models wrong" windows, shared vs model-specific confusions, the answer to Objective 3 |
| `7_conclusion.md` | M1 | 150 to 250 | Findings per objective, which model performed best on which criterion and why |
| `8_future_work.md` | M3 | 120 to 200 | Augmentation ablation, hybrid CNN-LSTM, cross-dataset tests, postural transitions, on-device deployment, more seeds |
| `acknowledgement.md` | M1 | 40 to 80 | Course and faculty thanks; dataset authors; **AI assistance disclosure** consistent with the README (Claude Code used for code and drafting; members reviewed and are responsible for their parts) |

Writing rules: no em dashes; no invented numbers or papers; "preliminary" wording only for interim results; the GRU is an ablation, not a fifth family.

### 6.3 Citation keys (`report/final/references.yaml`, M1)

| Key | Paper |
|---|---|
| `anguita2013` | Anguita et al., ESANN 2013 |
| `reyesortiz2016` | Reyes-Ortiz et al., Neurocomputing 2016 |
| `mahmud2020` | Mahmud et al., ECAI 2020 |
| `dirgova2022` | Dirgová Luptáková et al., Sensors 2022 |
| `yang2015` | Yang et al., IJCAI 2015 |
| `jiang2015` | Jiang and Yin, ACM MM 2015 |
| `ronao2016` | Ronao and Cho, ESWA 2016 |
| `wang2019` | Wang et al., PRL 2019 |
| `xia2020` | Xia et al., IEEE Access 2020 |
| `ordonez2016` | Ordóñez and Roggen, Sensors 2016 |
| `hammerla2016` | Hammerla et al., IJCAI 2016 |
| `murad2017` | Murad and Pyun, Sensors 2017 |
| `zhao2018` | Zhao et al., Math. Probl. Eng. 2018 |
| `ucihar2013` | Reyes-Ortiz et al., UCI HAR dataset, 2013 |
| `hochreiter1997` | Hochreiter and Schmidhuber, Neural Comput. 1997 |
| `schuster1997` | Schuster and Paliwal, IEEE TSP 1997 |
| `cho2014` | Cho et al., EMNLP 2014 |
| `vaswani2017` | Vaswani et al., NeurIPS 2017 |

18 references (Part C minimum is 12; all 13 interim papers overlap). In text: `[@ronao2016]`, or `[@ordonez2016; @hammerla2016]` for several. The builder numbers references in order of first citation and prints IEEE style `[3]` or `[4], [5]`.

### 6.4 Number placeholders (required for every result number)

- `{{final:<model>:<field>}}` → formatted from `results/final_summary.csv`. Fields: `val_macro_f1`, `test_acc`, `test_macro_f1`, `train_time_s`, `cpu_latency_bs1_ms`, `params`. Mean ± std where applicable, e.g. `0.941 ± 0.004`; `test_acc` as `93.9 ± 0.3%`. Add `.mean` or `.std` for a single value (`{{final:cnn1d:test_macro_f1.mean}}`).
- `{{analysis:<dotted.key>}}` → from `results/analysis/error_summary.json`.
- `{{tuning:<model>:<field>}}` → from the selected row of `results/tuning/<model>/tuning_summary.csv`.
- `{{split:<field>}}` → from `configs/split.json` (e.g. `val_subjects`).

The builder fails on any unresolved placeholder in a non-draft build. Dataset facts (128 samples, 50 Hz, 9 channels, 561 features, 30 subjects) may be written directly.

### 6.5 C.3 Individual Contribution Statement

The template says this page "must be written by the members themselves". So each member writes their own `report/final/contributions/<member>.md` personally (Claude Code may list that member's tasks and commits from git to jog their memory, but the words are theirs) and commits it under their own identity:
```markdown
---
model_task: "Model 2 (1D-CNN); evaluation harness; final results table and profiling"
---
<Key contributions in the member's own words, max 60 words.>
```
The builder fills the template's C.3 table (name, reg. no. from `members.local.yaml`, model/task, key contributions) and leaves Signature blank for hand signatures. Delete the unused fourth row.

### 6.6 Builder (`report/build_final_report.py`, M1, F1.4)

- Separate from `build_interim_report.py` (which must keep reproducing the interim report); import its shared helpers (markdown parsing, `Cursor`, table filling, member loading, PDF conversion, page counting).
- Start from a copy of the template: fill the cover (`Final report on`, names and reg. nos., month and year), copy the C.3 table, then delete all body content after the cover (Parts A, B and C instructions) **keeping the final `w:sectPr`**, and generate the paper content.
- **C.4 formatting:** Times New Roman throughout; A4; title 24 pt; author names 11 pt; affiliation 10 pt italic ("School of Computer Engineering, Manipal Institute of Technology, Manipal Academy of Higher Education, Manipal, India"); emails 9 pt Courier New; body 10 pt, justified, single column; level-1 headings small caps, centred, Roman numerals ("I. Introduction"); level-2 headings italic, left, lettered ("A. Model 1: MLP"); figures and tables centred; table captions above, 8 pt small caps, Roman numerals ("TABLE I. FINAL COMPARISON"); figure captions below, 8 pt, Arabic numerals ("Fig. 1. ..."); no page numbers, headers or footers. "Abstract:" and "Keywords:" as bold italic run-in labels (no em dash).
- **Generated tables:** Table I final comparison (`results/final_summary.csv`); Table II tuning grid and selected configuration per model (`results/tuning/*/tuning_summary.csv`); Table III per-class F1 (`results/analysis/per_class_f1.csv`).
- **Figures** (width about 3.3 in), placed in the section that discusses them: class distribution (V), final validation curves (V or VI-A), cost trade-off (VI-A), per-class F1 (VI-B), error overlap heatmap (VI-B), summed confusion matrix of the model with the best mean test macro-F1 (VI-B).
- Resolves citations (6.3) and placeholders (6.4); appends the numbered IEEE reference list; adds C.3 as the last page.
- Guards (non-draft): reject placeholder values in meta (empty, FILL_ME, TBD, `<...>`); fail on unresolved placeholders, unknown citation keys, missing section files, fewer than 12 cited references, or fewer than 10 papers cited in Section II; warn on any section outside its budget; fail on em dashes in inserted text. `--draft` turns failures into visible `[MISSING: ...]` markers.
- Sets document core properties (title, authors = the three members) instead of copying the template's metadata.
- Output: `report/build/ICT4442_Final_Report_HAR.docx` and `.pdf`; prints the page count (no page limit is stated for Part C; aim for about 7 to 9 pages including C.3).
- Tests in `tests/test_final_report_builder.py`: cover filling, Part A/B removal, sectPr kept, citation numbering, placeholder resolution and failure, C.3 table, formatting of headings and captions.

---

## 7. Turn order on the shared laptop

At every handover: the incoming member sets the **repo-local** identity (`git config user.name` / `user.email`, never `--global`), switches `gh` to their own account (`gh auth switch --user <login>`), and starts a fresh Claude Code session with their prompt. Each turn starts by reviewing and merging the previous member's open PR (D9), and ends with a handoff check (all committed, pushed, PR open, CI green, working tree clean).

| Turn | Member | Review first | Work |
|---|---|---|---|
| T1 | M1 | none | F1.1, F1.2 |
| T2 | M2 | M1's PRs | F2.1 |
| T3 | M3 (+ M1 approves too) | F2.1 PR: approved by M3 **and** M1 (D1 agreement), then merged | F3.1 |
| T4 | M1 | M3's PR | F1.3, then start F1.4 |
| T5 | M2 | M1's PRs | F2.2, F2.3 |
| T6 | M3 | M2's PR | F3.2, F3.3 |
| (checkpoint) | all | | All results exist: build the presentation from `results/final_summary.md`, `results/analysis/` and the figures |
| T7 | M1 | M3's PR | finish F1.4, F1.5 |
| T8 | M2 | M1's PR | F2.4 |
| T9 | M3 | M2's PR | F3.4 |
| T10 | M1 | M3's PR | Draft build; everyone reads it; owners fix their own sections in short follow-up turns |
| T11 | M3 | | F3.5 (reviewed and merged by M2) |
| T12 | M1 | | F1.6: final build, signatures, tag `final-v1`, submit |

---

## 8. CLAUDE.md amendment (appended by M1 in F1.1)

```markdown
## 7. Phase 3 amendments (agreed by all three; see docs/PHASE3_SPEC.md section 1)

- Final runs use seeds 42, 43 and 44, set only with `scripts/train.py --seed`; the allowed
  list is `final_seeds` in configs/base.yaml. These seeds are not protocol overrides.
- Tuning changes only `model.params`, using the grids in PHASE3_SPEC section 3, with
  validation only (`--no-test`). The shared training protocol is unchanged for all models.
- Every Phase 3 PR is approved by a different member, using their own gh account, before
  it is merged.
- Phase 3 tasks and the turn order are in docs/PHASE3_SPEC.md sections 2 and 7.
- The team shares one laptop: at each session start, confirm the repo-local git identity
  and the active gh account belong to the person at the keyboard before any commit.
```

---

## 9. Final-hour sequence and definition of done

In order:
1. All PRs merged; CI green; `pytest -q` passes.
2. `results/final_summary.md` regenerated after the last final run; 15 final run folders (5 models × 3 seeds); `results/profile_rtx4060-laptop.csv` and `results/analysis/` committed.
3. All section files and all three `contributions/*.md` committed by their owners; `docs/tasks/*.md` F rows up to date.
4. **M3:** regenerate `docs/contribution_log.md` (last commit before the build).
5. **M1:** `python report/build_final_report.py` (non-draft) succeeds; open the docx; check C.4 formatting, team number, every number against `results/final_summary.csv` and `results/analysis/`, and the reference list.
6. All three read the PDF and sign the C.3 page.
7. **M1:** `git tag -a final-v1 -m "Phase 3 final submission" && git push origin final-v1`; submit the PDF and the repo link.
