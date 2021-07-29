# COGN2021_Elipses
In this repository, code and results of paper "The Importance of Multiple Temporal Scalesin Motion Recognition:from Shallow to Deep Multi Scale Models" are reported.

The directory called "data" includes several txt files. In particular, features extracted using the splits reported in the following Figure which are input for the tested shallow Machine Learning model.
For each raw time series (velocity, pressure and radius) we extracted the list of features presented in the paper.

In the same directory, raw data, which represent the input for the tested Deep Learning Model, are also present.

The "Shallow" directory contains the tested shallow algorithms: Support Vector Machine (kernel Gaussian or linear), Random Forest, Naive Bayes.

The "Deep" directory contains the code of the tested deep algoritms: LSTM, the bidirectional LSTM and the code of the architecture proposed in the paper.
