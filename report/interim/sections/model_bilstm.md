---
model: bilstm
status: "Implemented, preliminary run"
notes: "2 bidirectional LSTM layers, final-state pooling; best epoch 22 of 32; 181 of 254 test errors are SITTING/STANDING"
ablations:
  gru:
    status: "Implemented, preliminary run"
    notes: "Same network with GRU cells (104,070 vs 138,502 parameters); best epoch 12 of 22; largest test confusion: SITTING/STANDING"
---
<!-- Written from results/bilstm/20261004-194812_134ae8c and results/gru/20261004-194905_134ae8c.
     Budget: 40 words or fewer. Covers the GRU ablation. -->
Two stacked bidirectional LSTM layers (64 units per direction) read the 128 x 9 window; the final forward and backward states feed a linear classifier. The GRU ablation swaps the cell to test whether the LSTM's extra gate matters.
