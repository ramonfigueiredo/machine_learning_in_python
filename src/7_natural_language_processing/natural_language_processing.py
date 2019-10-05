# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print("Predicting the Test set results\n", y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

'''
n = 200			Predicted (No)	Predicted (Yes)
Actual (No)		55				42
Actual (Yes)	12				91
'''
print("\n")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix\n", cm)

# Calculating metrics using the confusion matrix

print("\n")

TP = cm[0][0]
FN = cm[0][1]
TN = cm[1][0]
FP = cm[1][1]
print("True Positive (TP):", TP)
print("False Negative (FN):", FN)
print("True Negative (TN):", TN)
print("False Positive (FP):", FP)

print("\n")

accuracy = (TP + TN) / (TP + TN + FP + FN)
print("Accuracy = (TP + TN) / (TP + TN + FP + FN): %.2f %%" %(accuracy*100))

recall = TP / (TP + FN)
print("Recall = TP / (TP + FN): %.2f %%" %(recall*100) )

precision = TP / (TP + FP)
print("Precision = TP / (TP + FP): %.2f %%" %(precision*100) )

Fmeasure = (2 * recall * precision) / (recall + precision)
print("Fmeasure = (2 * recall * precision) / (recall + precision): %.2f %%" %(Fmeasure*100) )