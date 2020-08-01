# Emotion Classification on ISEAR Dataset

> University of Stuttgart

> CL Team Lab Project, Summer 2020

__This is the public accessible version of our work__

## Group Information
- Shawon Ashraf
- Christina Hitzl

## Project information

### Description
With emotion classification receiving more attention in the research field, the following paper deals with different approaches according to emotion detection. 
	The ISEAR dataset with seven emotions joy, anger, fear, shame, disgust, guilt and sadness was used for the previous research task.
	The main segmentation of this paper consists of two parts: on the one hand the reader will find frequency-based models and on the other hand the article describes weight-based methods.
	For this purpose, the implementation of the naïve Bayes and a lexicon-based approach with the NRC emotion lexicon, as well as a Multi label perceptron and CNN with embeddings serve as the basis for comparison. 


### Corpus
ISEAR - multi label dataset which can be availed from [SWISS CENTER FOR AFFECTIVE SCIENCES](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/)

| Labels  |
| ------- |
| anger   |
| joy     |
| shame   |
| fear    |
| guilt   |
| sadness |
| disgust |



## Setting up env

### Virtualenv
```bash
python -m venv venv
source venv/bin/activate
pip install -r ./config/requirements.txt
# or for windows
source .\venv\Scripts\activate
pip install -r .\config\requirements.txt
```

## Run
```bash
python ./scripts/main.py

# or for windows
python .\scripts\main.py
```
## CNN
The CNN is trained using [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings which can be downloaded from here
[http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip). Then the embeddings were retrofitted using Wordnet lexicon using the scripts provided by [Faruqui et al. (2015)](https://github.com/mfaruqui/retrofitting).

```bash
python ./scripts/main_cnn.py

# or for windows
python .\scripts\main_cnn.py
```

## Naive Bayes
```bash
python ./scripts/main_nb.py

# or for windows
python .\scripts\main_nb.py
```
