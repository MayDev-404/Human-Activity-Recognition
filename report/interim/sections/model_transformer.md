---
model: transformer
status: "Implemented, preliminary run"
notes: "3 pre-norm encoder layers; best epoch 33 of 43; largest test confusions: SITTING as STANDING, WALKING as WALKING_DOWNSTAIRS"
---
<!-- Written from results/transformer/20261004-192550_00d9884. Budget: 40 words or fewer. -->
A linear projection of the 9 channels to 64 dimensions, a learned positional embedding and three pre-norm self-attention layers, mean-pooled over the 128 steps. It tests whether attention over the whole window can replace recurrence.
