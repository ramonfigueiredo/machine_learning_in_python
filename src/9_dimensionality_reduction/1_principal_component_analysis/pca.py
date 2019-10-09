# PCA

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print("Predicting the Test set results\n", y_pred)

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix\n", cm)

'''
Confusion Matrix (cm)
 [[14  0  0]
 [ 1 15  0]
 [ 0  0  6]]

 c[0][0] c[0][1] c[0][2]
 c[1][0] c[1][1] c[1][2]
 c[2][0] c[2][1] c[2][2]

					   Predicted
			==============================
			|	Class1 	Class2 	Class3 	 |
A 			|----------------------------|
c 			|							 |
t 	Class1 	|	 14		  0		  0		 |
u 	Class2 	|	 1		  15	  0		 |
a 	Class3 	|	 0		  0		  6      |
l 			-----------------------------

PRECISION, RECALL, F1-SCORE FOR CLASS 1

accuracy_class1 = (TP_class1 + TN_class1) / sum_matrix_values
    TN_class1 = cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]

precision_class1 = TP_class1 / (cm[0][0] + cm[1][0] + cm[2][0])

recall_class1 = TP_class1 / (cm[0][0] + cm[0][1] + cm[0][2])

f1_score_class1 = (2 * recall_class1 * precision_class1) / (recall_class1 + precision_class1)


'''

# Showing classification report (Precision, Recall, F1-Score)
from sklearn.metrics import classification_report
print("Classification report\n", classification_report(y_test, y_pred, digits=3))

# Calculating metrics using the confusion matrix

print("\n")

TP_class1 = cm[0][0]
TP_class2 = cm[1][1]
TP_class3 = cm[2][2]
print("True Positive (TP) of class 1:", TP_class1)
print("True Positive (TP) of class 2:", TP_class2)
print("True Positive (TP) of class 3:", TP_class3)

sum_matrix_values = cm[0][0] + cm[0][1] + cm[0][2] + cm[1][0] + cm[1][1] + cm[1][2] + cm[2][0] + cm[2][1] + cm[2][2]

print("\nACCURACY, PRECISION, RECALL, F1-SCORE FOR CLASS 1\n")

TN_class1 = cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]
accuracy_class1 = (TP_class1 + TN_class1) / sum_matrix_values
print("Accuracy (class 1) = TP (class 1) + cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2] / sum_matrix_values: %.2f %%" %(accuracy_class1*100) )

precision_class1 = TP_class1 / (cm[0][0] + cm[1][0] + cm[2][0])
print("Precision (class 1) = TP (class 1) / (cm[0][0] + cm[1][0] + cm[2][0]): %.2f %%" %(precision_class1*100) )

recall_class1 = TP_class1 / (cm[0][0] + cm[0][1] + cm[0][2])
print("Recall (class 1) = TP (class 1) / (cm[0][0] + cm[0][1] + cm[0][2]): %.2f %%" %(recall_class1*100) )

f1_score_class1 = (2 * recall_class1 * precision_class1) / (recall_class1 + precision_class1)
print("F1-Score (class 1) = (2 * recall_class1 * precision_class1) / (recall_class1 + precision_class1): %.2f %%" %(f1_score_class1*100) )


print("\nnACCURACY, PRECISION, RECALL, F1-SCORE FOR CLASS 2\n")

TN_class2 = cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2]
accuracy_class2 = (TP_class2 + TN_class2) / sum_matrix_values
print("Accuracy (class 2) = TP (class 2) + cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2] / sum_matrix_values: %.2f %%" %(accuracy_class2*100) )

precision_class2 = TP_class2 / (cm[0][1] + cm[1][1] + cm[2][1])
print("Precision (class 2) = TP (class 2) / (cm[0][1] + cm[1][1] + cm[2][1]): %.2f %%" %(precision_class2*100) )

recall_class2 = TP_class2 / (cm[1][0] + cm[1][1] + cm[1][2])
print("Recall (class 2) = TP (class 2) / (cm[1][0] + cm[1][1] + cm[1][2]): %.2f %%" %(recall_class2*100) )

f1_score_class2 = (2 * recall_class2 * precision_class2) / (recall_class2 + precision_class2)
print("F1-Score (class 2) = (2 * recall_class2 * precision_class2) / (recall_class2 + precision_class2): %.2f %%" %(f1_score_class2*100) )


print("\nnACCURACY, PRECISION, RECALL, F1-SCORE FOR CLASS 3\n")

TN_class3 = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
accuracy_class3 = (TP_class3 + TN_class3) / sum_matrix_values
print("Accuracy (class 3) = TP (class 3) + cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1] / sum_matrix_values: %.2f %%" %(accuracy_class3*100) )

precision_class3 = TP_class3 / (cm[0][2] + cm[1][2] + cm[2][2])
print("Precision (class 3) = TP (class 3) / (cm[0][2] + cm[1][2] + cm[2][2]): %.2f %%" %(precision_class3*100) )

recall_class3 = TP_class3 / (cm[2][0] + cm[2][1] + cm[2][2])
print("Recall (class 3) = TP (class 3) / (cm[2][0] + cm[2][1] + cm[2][2]): %.2f %%" %(recall_class3*100) )

f1_score_class3 = (2 * recall_class3 * precision_class3) / (recall_class3 + precision_class3)
print("F1-Score (class 3) = (2 * recall_class3 * precision_class3) / (recall_class3 + precision_class3): %.2f %%" %(f1_score_class3*100) )

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()