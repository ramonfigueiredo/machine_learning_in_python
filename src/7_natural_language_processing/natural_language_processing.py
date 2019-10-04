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

'''
True Positive (TP) : Observation is positive, and is predicted to be positive.
False Negative (FN) : Observation is positive, but is predicted negative.
True Negative (TN) : Observation is negative, and is predicted to be negative.
False Positive (FP) : Observation is negative, but is predicted positive.
'''
print("\n")

TP = cm[0][0]
FN = cm[0][1]
TN = cm[1][0]
FP = cm[1][1]
print("True Positive (TP):", TP)
print("False Negative (FN):", FN)
print("True Negative (TN):", TN)
print("False Positive (FP):", FP)

'''
Metrics: 


1) Classification Rate/Accuracy: 
Classification Rate or Accuracy is given by the relation:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

However, there are problems with accuracy. 
It assumes equal costs for both kinds of errors. 
A 99% accuracy can be excellent, good, mediocre, poor or terrible depending upon the problem.


2) Recall:
Recall can be defined as the ratio of the total number of correctly classified positive examples divide to the total number of positive examples. 
High Recall indicates the class is correctly recognized (small number of FN).

Recall is given by the relation:

Recall = TP / (TP + FN)


3) Precision:
To get the value of precision we divide the total number of correctly classified positive examples by the total number of predicted positive examples. 
High Precision indicates an example labeled as positive is indeed positive (small number of FP).

Precision is given by the relation:

Precision = TP / (TP + FP)


High recall, low precision: 
This means that most of the positive examples are correctly recognized (low FN) but there are a lot of false positives.

Low recall, high precision: 
This shows that we miss a lot of positive examples (high FN) but those we predict as positive are indeed positive (low FP)


4) F-measure:
Since we have two measures (Precision and Recall) it helps to have a measurement that represents both of them. 
We calculate an F-measure which uses Harmonic Mean in place of Arithmetic Mean as it punishes the extreme values more.

The F-Measure will always be nearer to the smaller value of Precision or Recall.

Fmeasure = (2 * Recall * Precision) / (Recall + Presision)
'''
print("\n")

accuracy = (TP + TN) / (TP + TN + FP + FN)
print("Accuracy = (TP + TN) / (TP + TN + FP + FN): %.2f %%" %(accuracy*100))

recall = TP / (TP + FN)
print("Recall = TP / (TP + FN): %.2f %%" %(recall*100) )

precision = TP / (TP + FP)
print("Precision = TP / (TP + FP): %.2f %%" %(precision*100) )

Fmeasure = (2 * recall * precision) / (recall + precision)
print("Fmeasure = (2 * recall * precision) / (recall + precision): %.2f %%" %(Fmeasure*100) )