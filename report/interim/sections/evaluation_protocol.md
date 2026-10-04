<!-- Report section 2, shared evaluation protocol (M2). Budget: 100 words or fewer.
     Values come from configs/base.yaml (BUILD_SPEC 8.1); metrics from har.eval. -->
- Training: every model uses the same protocol from configs/base.yaml: Adam (learning rate 0.001), batch size 64, cross-entropy loss, gradient clipping at norm 1.0, at most 50 epochs with early stopping on validation macro-F1 (patience 10, best weights restored), seed 42, augmentation off.
- Metrics: accuracy, macro-F1, per-class precision, recall and F1, confusion matrices, trainable parameters, training time and single-thread CPU latency at batch size 1.
- Test discipline: model and run selection use validation only; the test split is evaluated once on each selected checkpoint.
- Timing: interim timings come from different machines; Phase 3 re-measures them on one.
