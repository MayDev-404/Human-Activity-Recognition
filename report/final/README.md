# Part C report sources

Inputs for `report/build_final_report.py` (PHASE3_SPEC section 6), which builds the final report into `report/build/` (gitignored).

| Path | What it holds | Owner |
|---|---|---|
| `meta.yaml` | Team number, title, month and year, affiliation, model order and table labels | M1 |
| `references.yaml` | IEEE reference strings, keyed by citation key | M1 |
| `sections/*.md` | One file per report section (PHASE3_SPEC 6.2) | Owner listed in 6.2 |
| `contributions/<member>.md` | Each member's C.3 contribution statement, written by that member (6.5) | Each member |

Section text follows the writing rules in PHASE3_SPEC 6.2: no em dashes, no invented numbers or papers, "preliminary" only for interim results, and the GRU is an ablation, not a fifth family.

## Building

```
python report/build_final_report.py --draft    # any stage: problems become visible [MISSING: ...] markers
python report/build_final_report.py            # final build: fails on any problem
```

Output: `report/build/ICT4442_Final_Report_HAR.docx` (`..._DRAFT.docx` for a draft) and a PDF when LibreOffice or Word is available; the builder prints the page count (aim for about 7 to 9 pages after the cover, including C.3). Section word budgets (PHASE3_SPEC 6.2) only print a `WARNING`; they never fail a build.

The C.4 author block needs each member's email. Emails stay out of the public repo: add them to the gitignored `report/interim/members.local.yaml`, next to the registration numbers:

```yaml
M1: "<reg no>"
M2: "<reg no>"
M3: "<reg no>"
emails: {M1: "<email>", M2: "<email>", M3: "<email>"}
```

Section files hold prose only: paragraphs and `-` bullets. The builder writes every heading itself (`I. Introduction`, `A. Model 1: MLP`), so headings inside a section file are dropped, and a markdown table in a section file fails the build (the result tables are generated). A contribution file has `model_task` in its front matter and at most 60 words of text.

## Citations

Cite a paper by its key from `references.yaml`:

- one paper: `[@ronao2016]`
- several papers: `[@ordonez2016; @hammerla2016]`

The builder numbers references in order of first citation, prints them IEEE style (`[3]`, or `[4], [5]` for several), and appends the numbered reference list. A non-draft build fails on an unknown key, on fewer than 12 cited references, or on fewer than 10 papers cited in Section II.

The values in `references.yaml` are copied verbatim from the verified list in `docs/BUILD_SPEC.md` section 9.2, and `tests/test_references.py` checks this. `*...*` in a value marks italics (journal and proceedings names). Never add a key for a paper whose DOI or URL has not been opened and checked (CLAUDE.md section 3). A new reference goes into BUILD_SPEC 9.2 first, then into `references.yaml`.

## Number placeholders

Every result number in a section file is a placeholder that the builder fills from a file in the repository. Never type a result by hand.

| Placeholder | Source | Example |
|---|---|---|
| `{{final:<model>:<field>}}` | `results/final_summary.csv` | `{{final:cnn1d:test_macro_f1}}` |
| `{{analysis:<dotted.key>}}` | `results/analysis/error_summary.json` | keys defined by the error analysis (F3.2) |
| `{{tuning:<model>:<field>}}` | the selected row of `results/tuning/<model>/tuning_summary.csv` | fields are that file's columns (F2.1) |
| `{{split:<field>}}` | `configs/split.json` | `{{split:val_subjects}}` |
| `{{table:<name>}}`, `{{fig:<name>}}` | the builder's numbering (below) | `{{table:final}}` gives `Table I` |

- `<model>` is one of `mlp`, `cnn1d`, `bilstm`, `gru`, `transformer`.
- `final` fields: `val_macro_f1`, `test_acc`, `test_macro_f1`, `train_time_s`, `cpu_latency_bs1_ms`, `params` (also `val_acc`). By default the builder prints mean ± sample standard deviation over the three seeds (for example `0.xxx ± 0.xxx`; `test_acc` as `xx.x ± x.x%`). Add `.mean` or `.std` for a single value: `{{final:cnn1d:test_macro_f1.mean}}`. The single-process CPU latency has no std, so it prints the mean alone.
- `tuning` fields: any column of `tuning_summary.csv`, for example `config_id`, a grid parameter such as `dropout`, `params`, `val_macro_f1` (`0.xxx`), `val_acc` (`xx.x%`), `best_epoch`.
- `analysis` keys: dotted paths into `error_summary.json` (a list index is a number, e.g. `overlap.pairs.0.jaccard`). Integers print as they are, other numbers with 3 decimals, lists comma-separated, and a `{mean, std}` pair as `mean ± std`. An optional format follows a `|`: `.0f` to `.3f` for the number of decimals, or `%` for a fraction as a percentage with one decimal. For example `{{analysis:overlap.all_wrong|.1f}}` gives `xx.x ± x.x` and `{{analysis:overlap.all_wrong_sitting_standing_share|%}}` gives `xx.x%`.
- `split` fields: `seed`, `n_val_subjects`, `val_subjects`, `train_subjects`, `test_subjects`.

## Tables and figures

The builder places and numbers them in this order, after the text of the section named. Refer to them with `{{table:<name>}}` and `{{fig:<name>}}` so the numbers always match.

| Placeholder | Number | Placed after | Source |
|---|---|---|---|
| `{{fig:class_distribution}}` | Fig. 1 | V (`5_setup.md`) | `docs/figures/class_distribution.png` |
| `{{table:final}}` | Table I | VI-A (`6a_results.md`) | `results/final_summary.csv` |
| `{{table:tuning}}` | Table II | VI-A | `results/tuning/<model>/tuning_summary.csv` |
| `{{fig:final_val_curves}}` | Fig. 2 | VI-A | `results/figures/final_val_curves.png` |
| `{{fig:cost_tradeoff}}` | Fig. 3 | VI-A | `results/figures/cost_tradeoff.png` |
| `{{table:per_class}}` | Table III | VI-B (`6b_error_analysis.md`) | `results/analysis/per_class_f1.csv` |
| `{{fig:per_class_f1}}` | Fig. 4 | VI-B | `results/analysis/per_class_f1.png` |
| `{{fig:error_overlap}}` | Fig. 5 | VI-B | `results/analysis/error_overlap.png` |
| `{{fig:confusion_best}}` | Fig. 6 | VI-B | `results/figures/confusion_test_<model>_final.png` of the model with the best mean test macro-F1 |

Table II sits in VI-A, after Table I, so that tables are numbered in order of appearance; `5_setup.md` refers to it as `{{table:tuning}}`. `per_class_f1.csv` (F3.2) may be long (columns `model`, `class`, and one column ending in `mean` and one ending in `std`) or wide (a `class` column plus `<model>_mean` and `<model>_std` columns).

A non-draft build fails on any placeholder it cannot resolve; `--draft` shows it as a visible `[MISSING: ...]` marker instead. Fixed dataset facts (128 samples per window, 50 Hz, 9 channels, 561 features, 30 subjects) may be written directly.
