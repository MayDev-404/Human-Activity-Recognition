<!-- III, M1. The three objectives of the synopsis, each tied to the measurement that answers it. Budget 80 to 140 words. -->
The project has three objectives, each answered by a fixed measurement on the official UCI HAR test subjects:

- **Performance under one protocol:** compare the MLP, 1D-CNN, bidirectional LSTM and Transformer encoder, with a GRU ablation of the recurrent model, by accuracy and macro-F1, each tuned with the same validation budget and reported as the mean and standard deviation over three seeds.
- **Accuracy versus cost:** relate that performance to the number of trainable parameters, the training time and the single-thread CPU latency at batch size 1, all measured on one machine, and identify the family best suited to on-device use.
- **Shared or architecture-specific errors:** determine whether the remaining errors are shared across the families, and therefore intrinsic to the sensor representation, or specific to some architectures, using per-class F1, the overlap of misclassified windows between models and their confusions.
