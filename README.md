# Spam Classification with Count Vectorizer and TF-IDF using Multinomial Naive Bayes
This repository contains a Python implementation of spam classification using the Count Vectorizer, TF-IDF, and Multinomial Naive Bayes approach. The goal of the project is to classify given messages as either spam or ham (non-spam) using machine learning techniques.

## Data
The data used in this project consists of a collection of SMS messages labeled as either spam or ham. The dataset is publicly available and can be found at [Link to dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset).


## Preprocessing
Before applying the machine learning algorithm, the data was preprocessed to clean and prepare it for modeling. This included removing punctuation, lowercasing all words, and removing stop words.

## Count Vectorizer and TF-IDF
The Count Vectorizer and TF-IDF approaches involve converting the messages into a matrix of token counts or TF-IDF values, respectively, where each column represents a unique word in the vocabulary and each row represents a message. The entries in the matrix are the count or TF-IDF value of each word in the corresponding message.


TF-IDF stands for Term Frequency-Inverse Document Frequency and is a weighting scheme that assigns higher weights to words that are more unique to a given document, and lower weights to words that are more common across all documents.


## Multinomial Naive Bayes
The Multinomial Naive Bayes classifier is a probabilistic model that makes predictions based on the probability of each word in a message belonging to the spam or ham class. It is called "naive" because it assumes that the presence of a word in a message is independent of the presence of other words in the same message.

## Modeling and Evaluation
The Count Vectorizer and TF-IDF approaches were used to transform the preprocessed data into feature matrices, which were then used to train and evaluate the Multinomial Naive Bayes model. The model was evaluated using common metrics such as precision, recall, and f1-score.
Conclusion

The results of the analysis show that the combination of Count Vectorizer/TF-IDF and Multinomial Naive Bayes is an effective approach for classifying spam and ham messages. The model achieved good performance on the test set, with high precision and recall scores.


## Packages:
- numpy
- pandas
- matplotlib
- seaborn
- nltk
- sklearn
