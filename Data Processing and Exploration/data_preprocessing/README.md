# Data Preprocessing
After receiving the data from HagaZiekenhuis, we needed to process it in a way such that the text could serve as input for our models. 

# Overview
This folder contains the following files:
	- get_labels_by_text Throws away duplicate texts and labels and makes sure all of the labels per row apply to the same text.
	- get_random_texts_to_label As we did not have the time to label all texts by hand we created this file which creates a random smaller subset.
	- get_texts_to_label We wanted to include edge cases in our hand-labelled text set, which is what this file does.
	