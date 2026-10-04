# Risk / Plan for Remaining Work

<!-- Shared file. Pull right before editing and change only your own heading.
     Draft content is in docs/BUILD_SPEC.md section 12; update it with what the
     preliminary results actually showed and remove any point they contradict.
     Budget for the whole section: 250 words plus the timeline table. -->

## Remaining work

<!-- M1: updated after the MLP and Transformer preliminary runs. -->
- All four models are implemented with one preliminary run each.
- Tune all models on validation only, with equal budgets.
- Final runs with seeds 42, 43 and 44 (mean and standard deviation).
- Error analysis: are SITTING/STANDING and stair-direction errors shared across architectures?

## Risks: Model 1 and 4 (M1)

<!-- M1: from runs mlp/20261004-192135 and transformer/20261004-192550. -->
- Transformer: sharp validation-to-test drop (largest error: SITTING as STANDING), possibly too little data for self-attention; any fix is chosen on validation only.
- MLP results mix architecture with the authors' engineered features; we state this when comparing it with raw-signal models.

## Risks: Model 2 (M2)

<!-- M2 (M2.8): from the preliminary cnn1d run; revisit after the Phase 3 seeds. -->
- Static postures: most CNN test errors are SITTING/STANDING confusions; the error analysis tests whether the MLP's gravity-angle features separate them better.
- Noisy selection: CNN validation macro-F1 swings between epochs, so the chosen epoch is partly chance; compare CNN settings only across the Phase 3 seeds.

## Risks: Model 3 (M3)

<!-- M3: from runs bilstm/20261004-194812 and gru/20261004-194905. -->
- Static postures: 181 of 254 BiLSTM test errors are SITTING/STANDING, which only the gravity channels separate.
- Cost: at batch size 1 on one CPU thread the GRU (6.225 ms) is slower than the BiLSTM (0.985 ms) because PyTorch accelerates only the LSTM kernel.

## Shared risks

<!-- M1 drafted these from BUILD_SPEC section 12; M2 and M3 may add bullets.
     Hardware: checked in the metrics.json of every run selected in results/summary.csv. -->
- Validation subjects may be easier than test subjects; we report both.
- Single seed: differences of about 1% are not meaningful yet.
- SITTING/STANDING confusion is expected from a waist-mounted phone; we analyse rather than tune it away.
- Fairness: identical split, preprocessing, optimiser and early stopping; the test split only reports selected checkpoints.
- Hardware: all interim runs used one laptop (RTX 4060 GPU, Ryzen 7 7840HS CPU), so interim timings are already same-machine; Phase 3 re-measures the final tuned models.

## Timeline to Final

| Dates | Milestone | Owner |
|---|---|---|
| 5 to 12 Oct | Tuning of all four models on validation (equal budget) | Each owner tunes their model |
| 13 to 19 Oct | Final 3-seed runs, comparison table, same-machine profiling | Each owner; M2 table and profiling |
| 20 to 25 Oct | Error analysis; Part C draft (one methodology subsection per owner) | All |
| 26 to 29 Oct | Presentation and viva | All |
| 31 Oct | Final report (Part C) and repository submission | All |
