# TV Commercial Classification Challenge

This challenge aims to classify TV news videos into two categories: commercials and non-commercials. Commercial blocks in news videos occupy almost 40-60% of total air time. The manual segmentation of commercials from thousands of TV news channels is time-consuming and economically infeasible. Hence, the need for machine learning-based methods arises. This challenge provides a dataset consisting of 150 hours of broadcast news videos from 5 different news channels, including 3 Indian and 2 international channels. The videos are recorded at a resolution of 720 X 576 at 25 fps using a DVR and set-top box.

## Objective

Given the audio-visual features extracted from each video shot, the objective is to classify each shot as a commercial or a non-commercial. This is a semantic video classification problem.

## Dataset

https://archive.ics.uci.edu/ml/machine-learning-databases/00326/

The dataset contains the following features extracted from each video shot:

Audio Features: Short term energy, zero crossing rate, spectral centroid, spectral flux, spectral roll-off frequency, fundamental frequency, and MFCC Bag of Audio Words.
Visual Features: Video shot length, screen text distribution, motion distribution, frame difference distribution, and edge change ratio.
The feature file is represented in Lib SVM data format and contains approximately 63% commercial instances. Dimension index for different features is provided in the data set information section.

## Evaluation Metric

The evaluation metric for this challenge is accuracy.

## Source

Dr. Prithwijit Guha , Raghvendra D. Kannao and Ravishankar Soni
Multimedia Analytics Lab,
Department of Electrical and Electronics Engineering,
Indian Institute of Technology, Guwahati, India
rdkannao '@' gmail.com , prithwijit.guha '@' gmail.com
