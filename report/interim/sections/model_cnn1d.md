---
model: cnn1d
status: "Implemented, preliminary run"
notes: "5 conv layers + global average pooling; most test errors are SITTING/STANDING confusions"
---
Five Conv1d, batch-norm and ReLU blocks with two max-pooling steps learn filters directly from the 128 x 9 raw window, then global average pooling. It tests whether learned local features can replace the 561 hand-crafted ones.
