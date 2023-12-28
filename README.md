# Exploring BERT Models for Dutch Clinical Lifestyle Classificarion: a thesis project
This repository contains the code for the creation and evaluation of several string matching and machine learning methods used for classification of Dutch clinical texts on the basis of the patient's smoking, alcohol usage and drugs usage statuses.
The data used in this project can not be provided due to privacy constraints.

# Overview
This repo contains the following subfolders:
```
└───src
│   └───Data Processing and Exploration (provides the code used for gathering, filtering and preparing the data used for pre-training and fine-tuning our models)
│   └───Model Training and Evaluation (provides the code to pre-train and fine-tune multiple BERT models, HAGALBERT is pre-trained from scratch, RobBERT-HAGA, belabBERT-HAGA and MedRoBERTa.nl-HAGA are further pre-trained on our data and BioBERT and ClinicalBERT are merely fine-tuned on translated input)
```