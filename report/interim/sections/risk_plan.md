# Risk / Plan for Remaining Work

<!-- Shared file. Pull right before editing and change only your own heading.
     Draft content is in docs/BUILD_SPEC.md section 12; update it with what the
     preliminary results actually showed and remove any point they contradict.
     Budget for the whole section: 250 words plus the timeline table. -->

## Remaining work

<!-- M1: revisit after the interim runs (e.g. if the Transformer stretch run happens). -->
- Train Model 4 (Transformer) under the shared protocol; report it even if weak.
- Tune all models on validation only, with an equal budget per model.
- Final runs with seeds 42, 43 and 44 (mean and standard deviation).
- Cost comparison (parameters, training time, CPU latency) on one machine.
- Error analysis: are SITTING/STANDING and stair-direction errors shared by all architectures or specific to some?

## Risks: Model 1 and 4 (M1)

<!-- M1: update with what the MLP run actually showed. -->
- Transformer data hunger: 15 training subjects is little data for self-attention; mitigated by a small pre-norm model with dropout (convolutional stem as an optional ablation).
- MLP results mix architecture with the authors' engineered features; we state this when comparing it with raw-signal models.

## Risks: Model 2 (M2)

<!-- M2 (M2.8) -->

## Risks: Model 3 (M3)

<!-- M3 (M3.9) -->

## Shared risks

<!-- M1 drafted these from BUILD_SPEC section 12; M2 and M3 may add bullets.
     Add a timing/hardware point once we know which machines the interim runs used. -->
- Validation subjects may be easier than test subjects; we report both.
- Single seed: differences of about 1% are not meaningful yet.
- SITTING/STANDING confusion is expected with a waist-mounted phone; we analyse it rather than tune it away.
- Fairness: identical split, preprocessing, optimiser and early stopping; the test split only reports selected checkpoints.

## Timeline to Final

| Dates | Milestone | Owner |
|---|---|---|
| 5 to 12 Oct | Model 4 trained; tuning of all models on validation | M1 (Model 4); each owner tunes their model |
| 13 to 19 Oct | Final 3-seed runs, comparison table, same-machine profiling | Each owner; M2 table and profiling |
| 20 to 25 Oct | Error analysis; Part C draft (one methodology subsection per owner) | All |
| 26 to 29 Oct | Presentation and viva | All |
| 31 Oct | Final report (Part C) and repository submission | All |
