---
model: transformer
status: "Implemented and unit-tested; training scheduled for week of 5 Oct"
notes: "3 pre-norm encoder layers, learned positional embedding, 159,302 parameters; not trained yet"
---
<!-- Update status, notes and the summary if the stretch run happens. Budget: 40 words or fewer. -->
A linear projection of the 9 channels to 64 dimensions, a learned positional embedding and three pre-norm self-attention layers, mean-pooled over the 128 steps. It tests whether attention over the whole window can replace recurrence.
