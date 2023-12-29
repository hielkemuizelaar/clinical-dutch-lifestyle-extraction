# Model Training and Evaluation
After finding out that query labels did not produce wanted results and that there was performance to be gained on string matching and standard machine learning we moved towards creating BERT models.

# Overview
This folder contains the following subfolders:
```
└───src
│   └───belabBERT-HAGA (Further pre-trains belabBERT on our full text set and fine-tunes it on our hand-labelled set.)
│   └───BioBERT (Fine-tunes BioBERT on our translated hand-labelled set.)
│   └───ClinicalBERT (Fine-tunes ClinicalBERT on our translated hand-labelled set.)
│   └───HAGALBERT (Pre-trains a new BERT model from scratch using our full text set.)
│   └───MedRoBERTa.nl-HAGA (Further pre-trains MedRoBERTa.nl on our full text set and fine-tunes it on our hand-labelled set.)
│   └───RobBERT-HAGA (Further pre-trains RobBERT on our full text set and fine-tunes it on our hand-labelled set.)
│   └───Translation of Input Texts (Translates the hand-labelled texts to English
│   └───Visualisation (Creates our t-SNE visualisations of BERT embeddings.)
```