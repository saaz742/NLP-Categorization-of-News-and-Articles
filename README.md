# NLP-Categorization-of-News-and-Articles

Dr Asgari 

Sharif university - Spring 2023

 - Sara Azarnoush

 - Mohammadreza Daviran

 - Nona Ghazizadeh

## Contents

  - [Document (Page 7)](https://github.com/saaz742/NLP-Categorization-of-News-and-Articles/blob/main/NLP_Spring1401_HW4.pdf)
  - [Document classification](https://github.com/saaz742/NLP-Categorization-of-News-and-Articles/blob/main/NLP_HW4_doc_clf.ipynb)
  - [Token classification HMM](https://github.com/saaz742/NLP-Categorization-of-News-and-Articles/blob/main/NLP_HW4_token_clf_HMM.ipynb)
  - [Token classification transformer](https://github.com/saaz742/NLP-Categorization-of-News-and-Articles/blob/main/NLP_HW4_token_clf_transformer.ipynb)
  - [Named Entity Recognition (NER) ](https://github.com/saaz742/NLP-Categorization-of-News-and-Articles/blob/main/NLP_HW4_NER%20(Extra).ipynb)

## Document classification
### Introduction
In this project, we classify persian_news documentation.
We import project requirements download the persian_news dataset preprocess it save clean data in JSON and read it. We use naive Bayes and transformers for classification and evaluate their result.
In naive Bayes, we use tf-idf to vectorize data and then use naive Bayes to predict the result.
In the transformer, we use SajjadAyoubi/distil-bigbird-fa-zwnj model to tokenize data make a new dataset, and train it to predict results.
More information is available in each section.

### Table of contents
 - Introduction
 - Requirements
    - Download
    - Import
 - Download and load
    - Download the persian_news dataset
    - Load dataset
 - Categories length
 - Preprocess
    - Preprocess
    - Save
      - Train
      - Validation
      - Test
 - Read data
 - Document classification
    - Naive Bayes
      - TF_IDF
      - Evaluate
    - Transformers
      - Model
      - Create Dataset
      - Train
      - Evaluate
        
## Token classification HMM
### Introduction
In this project, we Implement a model for extractive question and answer.
It inputs a text and question and returns an answer from the text.
We import project requirements and download the SajjadAyoubi/persian_qa dataset preprocess it save clean data in JSON and read it.
We made labels from the start index and end index of the answer in the text vectorized them with tf-idf and made a  new dataset.
More information is available in each section.

### Table of contents
 - Introduction
 - Requirements
    - Download
    - Import
 - Download and load
   - Load dataset
   - Example
 - Create dataset
 - Preprocess
  - Preprocess
  - Save
    - Train
    - Validation
    - Test
 - Read data
 - Labels
 - TF_IDF  
    - QADataset
    - TF_IDF
 - HMM
    - Vectorization
    - Model
    - Evaluation
        - F1 score
        - EM score
 - LSTM/CRF model
    - Requirements
    - Model
      - LSTM/CRF
      - Simple LSTM
      - Padding labels
    - Training
      - Training with training data
      - Training with training and validation data
      - Evaluation
 - Transformer
   
## Token classification transformer
### Introduction
In this project, we Implement a model for extractive question and answer.
It inputs a text and question and returns an answer from the text.
We import project requirements download the SajjadAyoubi/persian_qa dataset preprocess it and save the preprocessed data.
We make labels from the start index and end index of the answer in the text and use the HuggingFace pre-trained model to vectorize them and predict the answers for the related questions.
More information is available in each section.

### Table of contents
 - Introduction
 - Transformers
    - Requirements
    - Load Dataset
    - Preprocess
    - Model
        - Requirements
        - Fine Tune
    - Evaluation
        - EM and F1
        - Other Metrics
 - Inference
 - 
## Ner's 
### Introduction
In this project, we extract information to locate and classify named entities mentioned in unstructured text into pre-defined categories including 
Person (PER), Location (LOC), Main location (mainLoc), Event (EVE), Date (DAT), Organization (ORG), Time (TIM), Facility (FAC), Money (MON), Percent (PCT), Product (PRO). 
First, we import our requirements and download our dataset then preprocess data and translate our dataset labels to model labels and train our model. In the end, we check our true and predicted outputs and evaluate them. 
More information is available in each section.

### Table of contents
 - Introduction
 - Requirements
    - Download
    - Import
 - Prepare
   - Model
   - Data
 - Labels
    - Check
    - Translate
 - Predict
 - Calculate
 - Evaluation

