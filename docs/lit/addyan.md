| Paper (Author, Year) | Method | Dataset | Key Result | Relevance to Project |
|---|---|---|---|---|
| Ordóñez & Roggen, 2016 | DeepConvLSTM: four convolutional layers (64 filters) feeding two 128-cell LSTM layers | OPPORTUNITY, Skoda | Weighted F1 0.915 on OPPORTUNITY gestures (with Null), 0.930 on locomotion (without Null), 0.958 on Skoda | Widely used deep architecture for wearable HAR; recurrence on top of convolution helped, which motivates evaluating a recurrent model |
| Hammerla et al., 2016 | Over 4,000 experiments comparing MLP, CNN, LSTM and bidirectional LSTM | OPPORTUNITY, PAMAP2, Daphnet Gait | Bidirectional LSTM best on OPPORTUNITY (mean F1 0.745); CNN best on PAMAP2 (mean F1 0.937) | Closest earlier comparison of our model families; the best family depended on the activity type, the question we test on UCI HAR |
| Murad & Pyun, 2017 | Deep LSTM networks: unidirectional, bidirectional and cascaded variants | UCI HAR, USC-HAD, OPPORTUNITY, Daphnet FOG, Skoda | 96.7% accuracy on UCI HAR with a four-layer unidirectional LSTM | Strongest recurrent result reported on our dataset; an upper reference point for the BiLSTM and GRU |
| Zhao et al., 2018 | Residual bidirectional LSTM with three stacked layers | UCI HAR, OPPORTUNITY | UCI HAR: 93.6% accuracy and 93.5% F1; OPPORTUNITY: 90.5% F1 | Bidirectional LSTM evaluated on UCI HAR; the most direct published reference for our BiLSTM design |

## References
- F. J. Ordóñez and D. Roggen, "Deep convolutional and LSTM recurrent neural networks for multimodal wearable activity recognition," *Sensors*, vol. 16, no. 1, Art. no. 115, 2016, doi: 10.3390/s16010115.
- N. Y. Hammerla, S. Halloran, and T. Plötz, "Deep, convolutional, and recurrent models for human activity recognition using wearables," in *Proc. 25th Int. Joint Conf. Artif. Intell. (IJCAI)*, New York, NY, USA, 2016, pp. 1533-1540.
- A. Murad and J.-Y. Pyun, "Deep recurrent neural networks for human activity recognition," *Sensors*, vol. 17, no. 11, Art. no. 2556, 2017, doi: 10.3390/s17112556.
- Y. Zhao, R. Yang, G. Chevalier, X. Xu, and Z. Zhang, "Deep residual Bidir-LSTM for human activity recognition using wearable sensors," *Math. Probl. Eng.*, vol. 2018, Art. no. 7316954, 2018, doi: 10.1155/2018/7316954.
