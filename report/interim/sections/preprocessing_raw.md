<!-- Report section 2, raw-signal pipeline (M3). Budget: 180 words or fewer.
     Counts come from configs/split.json and docs/dataset_stats.md (scripts/data_report.py). -->
- Cleaning: the dataset is downloaded and verified automatically (all files and shapes); loading checks for missing or non-finite values and that signal, label and subject files align window by window.
- Encoding: labels 1 to 6 become class indices 0 to 5 for cross-entropy; no one-hot encoding is needed.
- Normalisation: per-channel z-score (mean and standard deviation over all windows and time steps), fitted on the 15 training subjects only and applied to every split.
- Split: the official subject split (21 training, 9 test subjects). Six training subjects (3, 15, 19, 22, 28, 29; seed 42) form the validation set, giving 5,276 training, 2,076 validation and 2,947 test windows. Windows overlap by 50%, so a random window-level split would put near-duplicates on both sides; splitting by subject prevents this leakage.
- Augmentation: jitter and per-channel scaling are implemented but disabled for all comparison runs.
- Input: 128 samples (2.56 s at 50 Hz) by 9 channels in a fixed order: body acceleration x, y, z; body angular velocity x, y, z; total acceleration x, y, z.
