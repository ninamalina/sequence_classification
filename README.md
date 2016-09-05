Classification of sequences to taxonomic classes with deep learning techniques.


Program is used for classifying bacterias to taxonomic ranks, based on their 16S rRNA sequences.

The 16S rRNA sequences have been downloaded from the RDP Ribosomal Database Project II repository. The preprocessed
dataset that we use contains 1000 randomly selected sequences from each of the three most common bacteria phyla,
Actinobacteria, Firmicutes, Proteobacteria - in total 3000 sequences.

We compare performance of 5 different models. The code generates random forest classifier, classification model using
convolutional neural network, classification model using recurrent neural network, classification model using bidirectional
recurrent neural network and a hybrid classification model that uses both convolutional and recurrent neural networks.
As a result we get two images, showing performance of given models in terms of accuracy and F1 score.