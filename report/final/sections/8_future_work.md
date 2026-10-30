<!-- VIII, M3. Budget 120 to 200 words. -->
- **Augmentation:** jitter and scaling are implemented but were switched off so that every model saw identical data; an ablation would show whether they narrow the gap between validation and test subjects.
- **Static postures:** sitting and standing dominate the errors all models share, so the raw-signal models could receive the phone's orientation to gravity as extra input channels, testing whether that explains the MLP's advantage.
- **Hybrid models:** convolution followed by recurrence [@ordonez2016; @xia2020] could combine the 1D-CNN's strength on the walking classes with sequence modelling.
- **Other datasets:** tests on other datasets and sensor placements would show whether the ranking holds beyond a waist-worn phone.
- **Postural transitions:** fixed windows ignore the transitions between postures that an extended version of the dataset labels [@reyesortiz2016].
- **On-device deployment:** latency was measured on a laptop CPU, and the GRU result shows that it depends on which kernels the runtime accelerates, so the models should be measured on a phone, also after quantisation.
- **More seeds:** the validation subjects did not predict the test ranking, so more seeds or cross-validation over subjects would make small differences interpretable.
