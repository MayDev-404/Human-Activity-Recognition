# Part C report sources

Inputs for `report/build_final_report.py` (PHASE3_SPEC section 6), which builds the final report into `report/build/` (gitignored).

| Path | What it holds | Owner |
|---|---|---|
| `references.yaml` | IEEE reference strings, keyed by citation key | M1 |
| `sections/*.md` | One file per report section (PHASE3_SPEC 6.2) | Owner listed in 6.2 |
| `contributions/<member>.md` | Each member's C.3 contribution statement, written by that member (6.5) | Each member |

Section text follows the writing rules in PHASE3_SPEC 6.2: no em dashes, no invented numbers or papers, "preliminary" only for interim results, and the GRU is an ablation, not a fifth family.

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

- `<model>` is one of `mlp`, `cnn1d`, `bilstm`, `gru`, `transformer`.
- `final` fields: `val_macro_f1`, `test_acc`, `test_macro_f1`, `train_time_s`, `cpu_latency_bs1_ms`, `params`. By default the builder prints mean ± sample standard deviation over the three seeds (for example `0.xxx ± 0.xxx`; `test_acc` as `xx.x ± x.x%`). Add `.mean` or `.std` for a single value: `{{final:cnn1d:test_macro_f1.mean}}`.
- `split` fields: `seed`, `n_val_subjects`, `val_subjects`, `train_subjects`, `test_subjects`.

A non-draft build fails on any placeholder it cannot resolve; `--draft` shows it as a visible `[MISSING: ...]` marker instead. Fixed dataset facts (128 samples per window, 50 Hz, 9 channels, 561 features, 30 subjects) may be written directly.
