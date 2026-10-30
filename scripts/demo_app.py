"""Live demo of the five final models, for the presentation.

Usage: streamlit run scripts/demo_app.py

Classifies windows from the 9 test subjects (never used for training or model selection) with the
seed-42 final checkpoint of every model, and shows the committed results and figures. Reported
numbers are read from files under results/ and configs/; the live check reruns the checkpoints on
the whole test split and compares the outcome with their run folders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from har.data.constants import ACTIVITY_NAMES, WINDOW_LEN
from har.data.splits import load_split
from har.demo import (DEMO_SEED, MODEL_ORDER, DemoModel, TestWindows, final_run_dir, load_demo_model,
                      load_test_windows, predict_proba, saved_predictions)
from har.eval.metrics import compute_metrics
from har.eval.plots import CATEGORICAL, GRID, INK, INK_SECONDARY, SEQUENTIAL, display_name
from har.utils.paths import CONFIGS, RESULTS_ROOT

SAMPLE_RATE_HZ = 50
CLASSES = [display_name(name) for name in ACTIVITY_NAMES]
MODEL_LABELS = {"mlp": "MLP", "cnn1d": "1D-CNN", "bilstm": "BiLSTM", "gru": "GRU", "transformer": "Transformer"}
INPUT_LABELS = {"features": "561 engineered features", "raw": "raw window, 128 × 9"}
ARCHITECTURES = {
    "mlp": "Fully connected layers on the dataset's hand-crafted features",
    "cnn1d": "Temporal convolutions, then global average pooling over time",
    "bilstm": "Bidirectional LSTM that reads the window forwards and backwards",
    "gru": "The same recurrent model with GRU cells (ablation)",
    "transformer": "Self-attention across all 128 time steps",
}
SIGNAL_GROUPS = (("Body acceleration (g)", 0), ("Angular velocity (rad/s)", 3), ("Total acceleration (g)", 6))
WINDOW_FILTERS = ("All windows", "Models disagree", "Every model is wrong")
FIGURES, ANALYSIS = RESULTS_ROOT / "figures", RESULTS_ROOT / "analysis"
PIPELINE_DOT = """
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#f4f4f1", color="#c3c2b7", fontname="Helvetica", fontsize=11];
  edge [color="#898781"];
  data [label="UCI HAR\\n30 subjects"]; split [label="Split by subject\\ntrain / val / test"];
  scale [label="Standardise\\n(training subjects only)"];
  feat [label="561 engineered\\nfeatures"]; raw [label="Raw window\\n128 steps × 9 channels"];
  mlp [label="MLP"]; cnn [label="1D-CNN"]; rnn [label="BiLSTM / GRU"]; tr [label="Transformer"];
  fit [label="Shared trainer\\nAdam, early stopping\\non val macro-F1"];
  test [label="Test once:\\naccuracy, macro-F1,\\nlatency"];
  data -> split -> scale; scale -> feat -> mlp; scale -> raw;
  raw -> cnn; raw -> rnn; raw -> tr;
  mlp -> fit; cnn -> fit; rnn -> fit; tr -> fit; fit -> test;
}
"""


@st.cache_resource(show_spinner="Loading the test windows and the five final checkpoints...")
def load_demo() -> tuple[TestWindows, dict[str, DemoModel], dict[str, np.ndarray]]:
    """Test windows, the seed-42 final models, and their probabilities ``(N, 6)`` on every test window."""
    windows = load_test_windows()
    models = {name: load_demo_model(final_run_dir(name)) for name in MODEL_ORDER}
    probs = {name: predict_proba(m.model, windows.inputs(m.representation)) for name, m in models.items()}
    return windows, models, probs


def signal_figure(signals: np.ndarray) -> Figure:
    """The 9 channels of one window ``(128, 9)``, one panel per sensor group."""
    t = np.arange(WINDOW_LEN) / SAMPLE_RATE_HZ
    fig = Figure(figsize=(6.4, 5.0), layout="constrained")
    axes = fig.subplots(3, 1, sharex=True)
    for ax, (title, start) in zip(axes, SIGNAL_GROUPS):
        for k, axis in enumerate("xyz"):
            ax.plot(t, signals[:, start + k], color=CATEGORICAL[k], linewidth=1.1, label=axis)
        ax.set_title(title, fontsize=9, loc="left", color=INK_SECONDARY)
        ax.grid(color=GRID, linewidth=0.5)
        ax.tick_params(labelsize=7, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(ncol=3, fontsize=7, frameon=False, loc="upper right")
    axes[-1].set_xlabel("Time (s)", fontsize=8)
    return fig


def probability_figure(probs: dict[str, np.ndarray], true: int) -> Figure:
    """Heatmap of each model's class probabilities ``(6,)`` for one window, true class outlined."""
    matrix = np.stack([probs[name] for name in MODEL_ORDER])
    fig = Figure(figsize=(6.0, 2.7), layout="constrained")
    ax = fig.subplots()
    ax.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CLASSES)), [c.replace(" ", "\n", 1) for c in CLASSES], fontsize=7)
    ax.set_yticks(range(len(MODEL_ORDER)), [MODEL_LABELS[n] for n in MODEL_ORDER], fontsize=8)
    for (i, j), p in np.ndenumerate(matrix):
        if p >= 0.005:
            ax.text(j, i, f"{p:.0%}", ha="center", va="center", fontsize=7, color="white" if p > 0.6 else INK)
    ax.add_patch(Rectangle((true - 0.5, -0.5), 1, len(MODEL_ORDER), fill=False,
                           edgecolor=CATEGORICAL[1], linewidth=2.5))
    ax.set_title("Predicted probability per class (orange outline: true activity)", fontsize=8,
                 loc="left", color=INK_SECONDARY)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def _pick_random(n: int) -> None:
    st.session_state.pos = int(np.random.default_rng().integers(n))


def live_tab(windows: TestWindows, models: dict[str, DemoModel], probs: dict[str, np.ndarray]) -> None:
    """Pick a test window, plot its signals and classify it with every model."""
    preds = np.stack([probs[name].argmax(axis=1) for name in MODEL_ORDER])   # (5, N)
    c1, c2, c3 = st.columns(3)
    subject = c1.selectbox("Test subject (never seen in training)",
                           ["Any", *sorted(np.unique(windows.subjects).tolist())])
    activity = c2.selectbox("True activity", ["Any", *CLASSES])
    which = c3.selectbox("Windows", WINDOW_FILTERS)

    mask = np.ones(len(windows.y), dtype=bool)
    if subject != "Any":
        mask &= windows.subjects == subject
    if activity != "Any":
        mask &= windows.y == CLASSES.index(activity)
    if which == WINDOW_FILTERS[1]:
        mask &= np.ptp(preds, axis=0) > 0
    elif which == WINDOW_FILTERS[2]:
        mask &= (preds != windows.y).all(axis=0)
    candidates = np.flatnonzero(mask)
    if len(candidates) == 0:
        st.info("No test windows match these filters.")
        return
    if st.session_state.get("pos", 0) >= len(candidates):
        st.session_state.pos = 0

    slider_col, button_col = st.columns([4, 1], vertical_alignment="bottom")
    button_col.button("Random window", on_click=_pick_random, args=(len(candidates),), width="stretch")
    if len(candidates) > 1:
        slider_col.slider(f"Window ({len(candidates):,} match)", 0, len(candidates) - 1, key="pos")
    idx = int(candidates[st.session_state.get("pos", 0)])
    true = int(windows.y[idx])

    # Classify the chosen window now, one forward pass per model.
    live = {name: predict_proba(m.model, windows.inputs(m.representation)[idx:idx + 1])[0]
            for name, m in models.items()}
    plot_col, pred_col = st.columns([1.1, 1])
    with plot_col:
        st.caption(f"Test window {idx}, subject {windows.subjects[idx]}: 2.56 s at {SAMPLE_RATE_HZ} Hz")
        st.pyplot(signal_figure(windows.signals[idx]))
    with pred_col:
        n_right = sum(int(p.argmax()) == true for p in live.values())
        m1, m2 = st.columns(2)
        m1.metric("True activity", CLASSES[true], border=True)
        m2.metric("Models correct", f"{n_right} of {len(MODEL_ORDER)}", border=True)
        table = pd.DataFrame([{
            "Model": MODEL_LABELS[name],
            "Input": INPUT_LABELS[models[name].representation],
            "Prediction": CLASSES[int(p.argmax())],
            "Confidence": 100 * float(p.max()),
            "Correct": "✅" if int(p.argmax()) == true else "❌",
        } for name, p in live.items()])
        st.dataframe(table, hide_index=True, column_config={
            "Confidence": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})
        st.pyplot(probability_figure(live, true))


def _pm(mean: pd.Series, std: pd.Series, scale: float = 1.0, fmt: str = ".3f", unit: str = "") -> list[str]:
    return [f"{m * scale:{fmt}} ± {s * scale:{fmt}}{unit}" for m, s in zip(mean, std)]


def results_tab(windows: TestWindows, models: dict[str, DemoModel], probs: dict[str, np.ndarray]) -> None:
    """The final results table and figures, plus a live check of the checkpoints."""
    summary = pd.read_csv(RESULTS_ROOT / "final_summary.csv")
    seeds = ", ".join(str(summary["seeds"].iloc[0]).split())
    best = summary.loc[summary["test_macro_f1_mean"].idxmax()]
    fastest = summary.loc[summary["cpu_latency_bs1_ms_mean"].idxmin()]
    smallest = summary.loc[summary["params"].idxmin()]
    c1, c2, c3 = st.columns(3)
    tile = {"delta_color": "off", "delta_arrow": "off", "border": True}   # delta names the model
    c1.metric("Best test macro-F1", f"{best['test_macro_f1_mean']:.3f}", MODEL_LABELS[best["model"]], **tile)
    c2.metric("Fastest on CPU (batch 1)", f"{fastest['cpu_latency_bs1_ms_mean']:.3f} ms",
              MODEL_LABELS[fastest["model"]], **tile)
    c3.metric("Fewest parameters", f"{int(smallest['params']):,}", MODEL_LABELS[smallest["model"]], **tile)

    st.subheader("Final results")
    st.caption(f"Mean ± sample std over seeds {seeds} (results/final_summary.csv). Settings were tuned on "
               "the validation subjects only; each run was evaluated on the test subjects once.")
    st.dataframe(pd.DataFrame({
        "Model": summary["model"].map(MODEL_LABELS),
        "Input": summary["representation"].map(INPUT_LABELS),
        "Parameters": summary["params"].map("{:,}".format),
        "Val macro-F1": _pm(summary["val_macro_f1_mean"], summary["val_macro_f1_std"]),
        "Test accuracy": _pm(summary["test_acc_mean"], summary["test_acc_std"], 100, ".1f", "%"),
        "Test macro-F1": _pm(summary["test_macro_f1_mean"], summary["test_macro_f1_std"]),
        "Train time (s)": _pm(summary["train_time_s_mean"], summary["train_time_s_std"], fmt=".1f"),
        "CPU latency, batch 1 (ms)": summary["cpu_latency_bs1_ms_mean"].map("{:.3f}".format),
    }), hide_index=True)

    c1, c2 = st.columns(2)
    c1.image(FIGURES / "cost_tradeoff.png", caption="Test macro-F1 against cost", width="stretch")
    c2.image(FIGURES / "final_val_curves.png", caption="Validation macro-F1 during training (seed 42)",
             width="stretch")
    name = st.selectbox("Test confusion matrix", MODEL_ORDER, format_func=MODEL_LABELS.get)
    st.image(FIGURES / f"confusion_test_{name}_final.png", width=620)

    st.subheader("Live check")
    st.caption(f"The app reran each seed-{DEMO_SEED} checkpoint on all {len(windows.y):,} test windows "
               "and compares the outcome with the files written when that run was trained.")
    rows = []
    for name in MODEL_ORDER:
        y_pred = probs[name].argmax(axis=1)
        live = compute_metrics(windows.y, y_pred, ACTIVITY_NAMES)
        recorded = models[name].metrics["test"]
        same = int((y_pred == saved_predictions(models[name].run_dir)).sum())
        rows.append({
            "Model": MODEL_LABELS[name],
            "Run": models[name].run_dir.name,
            "Live accuracy": f"{live['accuracy']:.2%}",
            "Recorded accuracy": f"{recorded['accuracy']:.2%}",
            "Live macro-F1": f"{live['macro_f1']:.4f}",
            "Recorded macro-F1": f"{recorded['macro_f1']:.4f}",
            "Same predictions as predictions_test.csv": f"{same:,} / {len(y_pred):,}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def errors_tab() -> None:
    """Committed error-analysis figures and summary (results/analysis/)."""
    st.subheader("Where the models go wrong")
    st.caption("From results/analysis/, over seeds 42, 43 and 44. To see these windows, choose "
               "'Every model is wrong' in the Live demo tab.")
    c1, c2 = st.columns(2)
    c1.image(ANALYSIS / "per_class_f1.png", caption="Per-class F1 on the test set", width="stretch")
    c2.image(ANALYSIS / "error_overlap.png", caption="Overlap of misclassified test windows between models",
             width="stretch")
    text = (ANALYSIS / "error_summary.md").read_text(encoding="utf-8")
    with st.expander("Full error analysis (results/analysis/error_summary.md)"):
        st.markdown("\n".join(line for line in text.splitlines() if not line.startswith("<!--")))


def method_tab(models: dict[str, DemoModel]) -> None:
    """Data, split, protocol and models, read from configs/ and the run folders."""
    split = load_split()
    base = yaml.safe_load((CONFIGS / "base.yaml").read_text(encoding="utf-8"))
    train, counts = base["train"], models["mlp"].metrics["counts"]
    st.subheader("Pipeline")
    st.graphviz_chart(PIPELINE_DOT, width="stretch")
    st.markdown(f"""
**Data.** UCI HAR: 30 volunteers with a smartphone at the waist doing 6 activities. Each sample is a
2.56 s window (128 readings at 50 Hz) of 9 inertial channels, plus 561 engineered features per window.

**Subject-independent split.** Train: {len(split['train_subjects'])} subjects ({counts['train']:,} windows).
Validation: subjects {', '.join(map(str, split['val_subjects']))} ({counts['val']:,} windows).
Test: subjects {', '.join(map(str, split['test_subjects']))} ({counts['test']:,} windows).
Nobody appears in more than one split, so every test prediction is about a person the model has never seen.

**Preprocessing.** Standardisation statistics are fitted on the training subjects only, then applied to
validation and test.

**Shared protocol** (configs/base.yaml). Adam, learning rate {train['lr']}, batch size {train['batch_size']},
cross-entropy, up to {train['max_epochs']} epochs with early stopping on validation macro-F1
(patience {train['patience']}), gradient clipping at {train['grad_clip_norm']}. Model settings were tuned
on validation only, then trained with seeds {', '.join(map(str, base['final_seeds']))}.
""")
    st.subheader("Models")
    st.dataframe(pd.DataFrame([{
        "Model": MODEL_LABELS[name],
        "Input": INPUT_LABELS[m.representation],
        "Architecture": ARCHITECTURES[name],
        "Tuned settings (configs/final)": ", ".join(
            f"{k}={v}" for k, v in yaml.safe_load((CONFIGS / "final" / f"{name}.yaml").read_text(
                encoding="utf-8"))["model"]["params"].items() if k not in ("n_classes", "in_channels", "input_dim")),
        "Parameters": f"{m.metrics['params_trainable']:,}",
    } for name, m in models.items()]), hide_index=True)


def sidebar() -> None:
    team = yaml.safe_load((CONFIGS / "team.yaml").read_text(encoding="utf-8"))["members"]
    st.sidebar.title("Human Activity Recognition")
    st.sidebar.caption("ICT 4442 Deep Learning mini project")
    st.sidebar.markdown("Four deep learning model families, trained under one shared, subject-independent "
                        "protocol, recognise activities from smartphone inertial sensor signals (UCI HAR).")
    st.sidebar.markdown("**Team**\n" + "\n".join(f"- {m['name']} ({m['id']})" for m in team))
    st.sidebar.caption(f"Live predictions use the seed-{DEMO_SEED} final checkpoint of each model, on CPU.")


def main() -> None:
    """Lay out the app: sidebar, then the live demo, results, error analysis and method tabs."""
    st.set_page_config(page_title="HAR model comparison", page_icon="📱", layout="wide")
    sidebar()
    try:
        windows, models, probs = load_demo()
    except FileNotFoundError as err:
        st.error(f"{err}\n\nThe dataset comes from `python -m har.data.download`; the checkpoints exist only "
                 "on the machine that trained the final runs.")
        st.stop()
    live, results, errors, method = st.tabs(["Live demo", "Results", "Error analysis", "How it works"])
    with live:
        live_tab(windows, models, probs)
    with results:
        results_tab(windows, models, probs)
    with errors:
        errors_tab()
    with method:
        method_tab(models)


if __name__ == "__main__":
    main()
