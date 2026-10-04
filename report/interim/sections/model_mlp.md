---
model: mlp
status: "Implemented, preliminary run"
notes: "2 hidden layers (256, 128); best epoch 10 of 20; largest test confusion: SITTING/STANDING"
---
<!-- Written from results/mlp/20261004-192135_7a7e7a3. Budget: 40 words or fewer. -->
Two hidden layers (256 and 128 units, batch normalisation, dropout 0.3) on the 561 engineered features. It is the feed-forward baseline: it shows how far the dataset authors' hand-crafted features go without any sequence modelling.
