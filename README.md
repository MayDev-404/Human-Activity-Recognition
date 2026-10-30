# Comparative Evaluation of Deep Learning Architectures for Human Activity Recognition

ICT 4442 Deep Learning mini project, School of Computer Engineering, MIT Manipal.

We compare four deep learning model families for human activity recognition from smartphone inertial sensor signals, all trained and evaluated under one shared, subject-independent protocol on the UCI HAR dataset:

| Model | Input | Owner |
|---|---|---|
| 1. MLP | 561 engineered features | M1 |
| 2. 1D-CNN | raw windows, 128 time steps x 9 channels | M2 |
| 3. BiLSTM, with a GRU ablation | raw windows, 128 x 9 | M3 |
| 4. Transformer encoder | raw windows, 128 x 9 | M1 |

## Team

| ID | Member | GitHub | Owns |
|---|---|---|---|
| M1 | Mayank Kejariwal | [MayDev-404](https://github.com/MayDev-404) | MLP, Transformer, engineered-feature pipeline, repository scaffold, report assembly |
| M2 | Ishan Abhijit Saraf | [IshanCodes21](https://github.com/IshanCodes21) | 1D-CNN, shared evaluation harness (metrics, trainer, train script, profiling, results table) |
| M3 | Addyan Kumar | [Addhyanitis](https://github.com/Addhyanitis) | BiLSTM and GRU ablation, dataset download, raw-signal loading, normalisation, validation split, augmentation |

Each member commits their own work under their own GitHub account. The full plan and file ownership are in [docs/BUILD_SPEC.md](docs/BUILD_SPEC.md), and task status is in [docs/tasks/](docs/tasks/).

## Dataset

UCI HAR, "Human Activity Recognition Using Smartphones" (UCI id 240): 30 volunteers wearing a smartphone at the waist, 6 activities (walking, walking upstairs, walking downstairs, sitting, standing, laying), pre-segmented into 2.56 s windows (128 samples at 50 Hz, 50% overlap). Each window has 9 raw inertial channels and 561 engineered time and frequency domain features.

We use the official subject-independent split (21 training subjects with 7,352 windows, 9 test subjects with 2,947 windows) and hold out 6 of the training subjects for validation, chosen once with seed 42 and stored in `configs/split.json`.

The dataset is published under the Creative Commons Attribution 4.0 (CC BY 4.0) licence. It is downloaded into `data/` by `python -m har.data.download` and is not redistributed in this repository. If you use it, cite:

- J. Reyes-Ortiz, D. Anguita, A. Ghio, L. Oneto, and X. Parra, "Human Activity Recognition Using Smartphones," UCI Machine Learning Repository, 2013, doi: 10.24432/C54S4K.
- D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, "A public domain dataset for human activity recognition using smartphones," in *Proc. 21st Eur. Symp. Artif. Neural Netw., Comput. Intell. Mach. Learn. (ESANN)*, Bruges, Belgium, 2013, pp. 437-442.

## Setup

Python 3.10 to 3.13.

```bash
git clone https://github.com/MayDev-404/Human-Activity-Recognition.git
cd Human-Activity-Recognition
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch **first**, otherwise pip may install the CPU build and then treat the CUDA build as already satisfied:

- NVIDIA GPU: use the command for your CUDA version from https://pytorch.org/get-started/locally/, then check `python -c "import torch; print(torch.cuda.is_available())"`.
- CPU only: `pip install torch`.

Then install the rest and run the tests (tests that need the dataset skip automatically until it is downloaded):

```bash
pip install -r requirements.txt
pip install -e .
pytest -q
```

## How to run

```bash
python -m har.data.download                                  # dataset into data/raw/
python scripts/train.py --config configs/mlp.yaml --smoke    # quick end-to-end check, written to results/_smoke/ (not committed)
python scripts/train.py --config configs/mlp.yaml            # full run, written to results/mlp/<run_id>/
python scripts/make_results_table.py                         # results/summary.csv, results/summary.md and the block below
```

### Live demo

```bash
streamlit run scripts/demo_app.py
```

Opens a local web app (http://localhost:8501) with four tabs: a live demo that classifies any window from the 9 test subjects with the seed-42 final checkpoint of every model, the final results table and figures, the error analysis, and a summary of the method. A live check reruns each checkpoint on the whole test split and compares the outcome with its run folder. The checkpoints (`best.pt`) are gitignored, so the app runs only on the machine that trained the final runs.

Protocol, shared by every model and stored only in `configs/base.yaml`: Adam (lr 1e-3), batch size 64, cross-entropy loss, at most 50 epochs with early stopping on validation macro-F1 (patience 10, best weights restored), gradient clipping at norm 1.0, seed 42, augmentation off. Add `--no-test` while debugging. Models and runs are selected on the validation split only; the test split is evaluated once, on the selected checkpoint.

## Results

Preliminary, untuned, single-seed results. This block is generated by `scripts/make_results_table.py` from the run folders under `results/`; do not edit it by hand.

<!-- RESULTS:START -->
<!-- Generated by scripts/make_results_table.py from the run folders tagged final. Do not edit by hand. -->
Final results: mean ± sample std over seeds 42, 43, 44 of the validation-selected configuration; shared protocol.

| Model | Run by | Representation | Params | Val Macro-F1 | Test Acc | Test Macro-F1 | Train time (s) | CPU latency bs=1 (ms) | Seeds |
|---|---|---|---|---|---|---|---|---|---|
| mlp | Mayank Kejariwal | features | 212,230 | 0.956 ± 0.008 | 94.1 ± 0.4% | 0.939 ± 0.003 | 8.6 ± 1.8 | 0.087 (single process) | 42, 43, 44 |
| cnn1d | Ishan Abhijit Saraf | raw | 197,702 | 0.970 ± 0.000 | 93.0 ± 0.9% | 0.931 ± 0.009 | 10.3 ± 3.2 | 0.673 (single process) | 42, 43, 44 |
| bilstm | Addyan Kumar | raw | 138,502 | 0.955 ± 0.012 | 90.6 ± 1.2% | 0.907 ± 0.012 | 9.5 ± 2.4 | 1.439 (single process) | 42, 43, 44 |
| gru | Addyan Kumar | raw | 104,070 | 0.970 ± 0.001 | 90.6 ± 1.3% | 0.905 ± 0.015 | 7.1 ± 3.0 | 9.328 (single process) | 42, 43, 44 |
| transformer | Mayank Kejariwal | raw | 109,766 | 0.960 ± 0.003 | 89.8 ± 1.2% | 0.897 ± 0.012 | 16.4 ± 4.9 | 1.745 (single process) | 42, 43, 44 |

CPU latency (batch size 1, single thread) marked (single process) comes from one run of scripts/profile_all.py on rtx4060-laptop (results/profile_rtx4060-laptop.csv). All final runs report the same hardware (GPU NVIDIA GeForce RTX 4060 Laptop GPU, CPU AMD Ryzen 7 7840HS w/ Radeon 780M Graphics).
<!-- RESULTS:END -->
