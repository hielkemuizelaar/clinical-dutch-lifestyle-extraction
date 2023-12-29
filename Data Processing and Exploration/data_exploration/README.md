# Data Exploration
One of our research questions entailed figuring out if string matching and "standard" machine learning would already provide satisfying results on the task-at-hand, which would nullify the need for deeper BERT architectures. In this folder we explored our data and conducted these tests. 

# Overview
This folder contains the following subfolders:
```
└───src
│   └───feature_exploration (Contains code used to gain insight in the data)
│   └───hand_labels (Contains code for testing our string matching and standard machine learning models on hand-labelled input, rather than input that was query-labelled.)
```

This folder contains the following files:
	- add_query_labels_to_hand_text Aligns the query labels and the hand assigned labels to the same text for the purpose of comparison.
	- New Drinking Experiment 1 ... New Smoking Experiment 2 Are experiments with query labels performed on each of the three lifestyle tasks. 