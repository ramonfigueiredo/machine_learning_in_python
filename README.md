Machine Learning in Python
===========================

## Contents
1. [Data Preprocessing](#data-preprocessing)
2. [Regression](#regression)
	1. [Simple Linear Regression](#simple-linear-regression)
	2. [Multiple Linear Regression](#multiple-linear-regression)
	3. [Polynomial Regression](#polynomial-regression)
	4. [Support Vector Regression](#support-vector-regression)
	5. [Decision Tree Regressor](#decision-tree-regressor)
	6. [Random Forest Regression](#random-forest-regression)
3. [Classification](#classification)
	1. [Logistic Regression](#logistic-regression)
	2. [K-Nearest Neighbors](#k-nearest-neighbors)
	3. [Support Vector Machine](#support-vector-machine)
	4. [Kernel SVM](#kernel-svm)
	5. [Naive Bayes](#naive-bayes)
	6. [Decision Tree Classification](#decision-tree-classification)
	7. [Random Forest Classification](#random-forest-classification)
4. [Clustering](#clustering)
	1. [K-Means Clustering](#k-means-clustering)
4. [How to run the Python program](#how-to-run-the-python-program)

## Data Preprocessing

### Data Preprocessing

a.  [data_preprocessing.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/1_data_preprocessing/data_preprocessing.py)

* Taking care of missing data
* Encoding categorical data
* Splitting the dataset into the Training set and Test set
* Feature Scaling

Go to [Contents](#contents)

## Regression

| Regression Model 				  |	Pros 																					 | Cons |
| ------------------------------- |	---------------------------------------------------------------------------------------- | ---- |
| Linear Regression 			  | Works on any size of dataset, gives informations about relevance of features 			 | The Linear Regression Assumptions |
| Polynomial Regression 		  | Works on any size of dataset, works very well on non linear problems 					 | Need to choose the right polynomial degree for a good bias/variance tradeoff |
| Support Vector Regression (SVR) | Easily adaptable, works very well on non linear problems, not biased by outliers 		 | Compulsory to apply feature scaling, not well known, more difficult to understand |
| Decision Tree Regression  	  | Interpretability, no need for feature scaling, works on both linear / nonlinear problems | Poor results on too small datasets, overfitting can easily occur |
| Random Forest Regression 		  | Powerful and accurate, good performance on many problems, including non linear | No interpretability, overfitting can easily occur, need to choose the number of trees |

### Simple Linear Regression

a.  [simple_linear_regression.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/1_simple_linear_regression/simple_linear_regression.py)

* Importing the dataset ([Salary_Data.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/1_simple_linear_regression/Salary_Data.csv))
* Splitting the dataset into the Training set and Test set
* Fitting Simple Linear Regression to the Training set
* Predicting the Test set results
* Visualising the Training and Test set results

* Visualising the Training set results
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/1_simple_linear_regression/Visualising-the-Test-set-results.png)
* Visualising the Test set results
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/1_simple_linear_regression/Visualising-the-Test-set-results.png)

Go to [Contents](#contents)

### Multiple Linear Regression

a.  [multiple_linear_regression.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/2_multiple_linear_regression/multiple_linear_regression.py)

b. Multiple Linear Regression - Automatic Backward Elimination with p-values only: [backward_elimination_with_p_values_only.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/2_multiple_linear_regression/backward_elimination_with_p_values_only.py)

c. Multiple Linear Regression - Automatic Backward Elimination with p-values and adjusted R-squared: [backward_elimination_with_p_values_and_adjusted_r_squared.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/2_multiple_linear_regression/backward_elimination_with_p_values_and_adjusted_r_squared.py)

* Importing the dataset ([50_Startups.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/2_multiple_linear_regression/50_Startups.csv))
* Encoding categorical data
* Avoiding the Dummy Variable Trap
* Splitting the dataset into the Training set and Test set
* Fitting Multiple Linear Regression to the Training set
* Predicting the Test set results
* Building the optimal model using Backward Elimination

Go to [Contents](#contents)

### Polynomial Regression

a.  [polynomial_regression.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/3_polynomial_regression/polynomial_regression.py)

* Importing the dataset ([Position_Salaries.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/3_polynomial_regression/Position_Salaries.csv))
* Fitting Linear Regression to the Training set
* Predicting a new result with Linear Regression
* Visualising the Linear Regression results
* Fitting Polynomial Regression (degree = 2, 3, and 4) to the Training set
* Predicting a new result with Polynomial Regression (degree = 2, 3, and 4)
* Visualising the Polynomial Regression (degree = 2, 3, and 4) results (for higher resolution and smoother curve)

* Visualising the Linear Regression results
![Visualising the Linear Regression results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/3_polynomial_regression/Truth_or_Bluff-Linear_Regression.png)
* Visualising the Polynomial Regression results (degree = 2)
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/3_polynomial_regression/Truth_or_Bluff-Polynomial_Regression-degree_2.png)
* Visualising the Polynomial Regression results (degree = 3)
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/3_polynomial_regression/Truth_or_Bluff-Polynomial_Regression-degree_3.png)
* Visualising the Polynomial Regression results (degree = 4)
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/3_polynomial_regression/Truth_or_Bluff-Polynomial_Regression-degree_4.png)

Go to [Contents](#contents)

### Support Vector Regression

a.  [svr.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/4_support_vector_regression/svr.py)

* Importing the dataset ([Position_Salaries.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/4_support_vector_regression/Position_Salaries.csv))
* Feature Scaling
* Fitting Support Vector Regression (SVR) to the dataset
* Predicting a new result with Support Vector Regression (SVR)
* Visualising the SVR results (for higher resolution and smoother curve)

![Visualising the SVR results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/4_support_vector_regression/Visualising-the-SVR-results.png)

Go to [Contents](#contents)

### Decision Tree Regressor

a.  [decision_tree_regression.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/5_decision_tree_regression/decision_tree_regression.py)

* Importing the dataset ([Position_Salaries.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/5_decision_tree_regression/Position_Salaries.csv))
* Fitting Decision Tree Regression to the dataset
* Predicting a new result with Decision Tree Regression
* Visualising the Decision Tree Regression results (higher resolution)

![Visualising the Decision Tree Regression results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/5_decision_tree_regression/Visualising-the-Decision-Tree-Regression-results.png)

Go to [Contents](#contents)

### Random Forest Regression

a.  [random_forest_regression.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/6_random_forest_regression/random_forest_regression.py)

* Importing the dataset ([Position_Salaries.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/6_random_forest_regression/Position_Salaries.csv))
* Fitting Random Forest Regression to the dataset
* Predicting a new result with Random Forest Regression
* Visualising the Random Forest Regression results (higher resolution)

![Visualising the Random Forest Regression results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/6_random_forest_regression/Visualising-the-Random-Forest-Regression-results.png)

Go to [Contents](#contents)

## Classification

### Logistic Regression

a.  [logistic_regression.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/1_logistic_regression/logistic_regression.py)

* Importing the dataset ([Social_Network_Ads.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/1_logistic_regression/Social_Network_Ads.csv))
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Fitting Logistic Regression to the Training set
* Predicting the Test set results with Logistic Regression
* Making the Confusion Matrix
* Visualising the Training set results
* Visualising the Test set results

![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/1_logistic_regression/Visualising-the-Training-set-results.png)

![Visualising-the-Test-set-results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/1_logistic_regression/Visualising-the-Test-set-results.png)

Go to [Contents](#contents)

### K-Nearest Neighbors

a.  [knn.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/2_k_nearest_neighbors/knn.py)

* Importing the dataset ([Social_Network_Ads.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/2_k_nearest_neighbors/Social_Network_Ads.csv))
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Fitting K-NN to the Training set
* Predicting the Test set results
* Making the Confusion Matrix
* Visualising the Training set results
* Visualising the Test set results

![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/2_k_nearest_neighbors/Visualising-the-Training-set-results.png)

![Visualising-the-Test-set-results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/2_k_nearest_neighbors/Visualising-the-Test-set-results.png)

Go to [Contents](#contents)

### Support Vector Machine

a.  [svm.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/3_support_vector_machine/svm.py)

* Importing the dataset ([Social_Network_Ads.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/3_support_vector_machine/Social_Network_Ads.csv))
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Fitting SVM to the Training set
* Predicting the Test set results
* Making the Confusion Matrix
* Visualising the Training set results
* Visualising the Test set results

![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/3_support_vector_machine/Visualising-the-Training-set-results.png)

![Visualising-the-Test-set-results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/3_support_vector_machine/Visualising-the-Test-set-results.png)

Go to [Contents](#contents)

### Kernel SVM

a.  [kernel_svm.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/4_kernel_svm/kernel_svm.py)

* Importing the dataset ([Social_Network_Ads.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/4_kernel_svm/Social_Network_Ads.csv))
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Fitting Kernel SVM to the Training set
* Predicting the Test set results
* Making the Confusion Matrix
* Visualising the Training set results
* Visualising the Test set results

![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/4_kernel_svm/Visualising-the-Training-set-results.png)

![Visualising-the-Test-set-results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/4_kernel_svm/Visualising-the-Test-set-results.png)

Go to [Contents](#contents)

### Naive Bayes

a.  [naive_bayes.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/5_naive_bayes/naive_bayes.py)

* Importing the dataset ([Social_Network_Ads.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/5_naive_bayes/Social_Network_Ads.csv))
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Fitting Naive Bayes to the Training set
* Predicting the Test set results
* Making the Confusion Matrix
* Visualising the Training set results
* Visualising the Test set results

![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/5_naive_bayes/Visualising-the-Training-set-results.png)

![Visualising-the-Test-set-results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/5_naive_bayes/Visualising-the-Test-set-results.png)

Go to [Contents](#contents)

### Decision Tree Classification

a.  [decision_tree_classification.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/6_decision_tree_classification/decision_tree_classification.py)

* Importing the dataset ([Social_Network_Ads.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/6_decision_tree_classification/Social_Network_Ads.csv))
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Fitting Decision Tree Classification to the Training set
* Predicting the Test set results
* Making the Confusion Matrix
* Visualising the Training set results
* Visualising the Test set results

![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/6_decision_tree_classification/Visualising-the-Training-set-results.png)

![Visualising-the-Test-set-results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/6_decision_tree_classification/Visualising-the-Test-set-results.png)

Go to [Contents](#contents)

### Random Forest Classification

a.  [random_forest_classification.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/7_random_forest_classification/random_forest_classification.py)

* Importing the dataset ([Social_Network_Ads.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/7_random_forest_classification/Social_Network_Ads.csv))
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Fitting Random Forest Classification to the Training set
* Predicting the Test set results
* Making the Confusion Matrix
* Visualising the Training set results
* Visualising the Test set results

![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/7_random_forest_classification/Visualising-the-Training-set-results.png)

![Visualising-the-Test-set-results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/3_classification/7_random_forest_classification/Visualising-the-Test-set-results.png)

Go to [Contents](#contents)

## Clustering

### K-Means Clustering
* Importing the dataset ([Mall_Customers.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/1_k_means_clustering/Mall_Customers.csv))
* Using the [Elbow method](https://en.wikipedia.org/wiki/Elbow_method_(clustering)) to find the optimal number of clusters
	* The Elbow method is a heuristic method of interpretation and validation of consistency within cluster analysis designed to help finding the appropriate number of clusters in a dataset
* Using k-means++ to select initial cluster centers for k-mean clustering in a smart way to speed up convergence
* Plotting the Elbow method
	* The Elbow method uses the Within-Cluster Sum of Squares (WCSS) metric = Sum of squared distances of samples to their closest cluster center.
	* According to the Elbow method the best number of cluster in the mall customers dataset ([Mall_Customers.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/1_k_means_clustering/Mall_Customers.csv)) is 5
* Fitting K-Means to the dataset. The fit method returns for each observation which cluster it belongs to.
* Visualising the clusters
	* Cluster 1 has high income and low spending score. A better name for this cluster of clients as "Careful clients"
	* Cluster 2 has average income and average spending score. A better name for this cluster of clients as "Standard clients"
	* Cluster 3 has high income and high spending score. A better name for this cluster of clients as "Target clients"
		* Therefore, cluster 3 is the cluster of clients that would be the main potential target of the mall marketing campaigns and it would be very insighful for them all to understand what kind of products are bought by the clients in this cluster 
	* Cluster 4 has low income and low spending score. A better name for this cluster of clients as "Careless clients" 
	* Cluster 5 has low income and low spending score. A better name for this cluster of clients as "Sensible clients"

a.  [kmeans.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/1_k_means_clustering/kmeans.py)

![The Elbow Method](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/1_k_means_clustering/The-Elbow-Method.png)

![Clusters of customers](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/1_k_means_clustering/Clusters-of-customers.png)

Go to [Contents](#contents)

## How to run the Python program?

1. Install [virtualenv](https://virtualenv.pypa.io/en/latest/)
	* To activate the virtualenv on Linux or MacOS: ```source venv/bin/activate```
	* To activate the virtualenv on Windows: ```\venv\Script\activate.bat```

2. Run the program

```sh
cd <folder_name>/

virtualenv venv

source venv/bin/activate

pip install -r requirements.txt

python <name_of_python_program>.py
```

**Note**: To desactivate the virtual environment

```sh
deactivate
```

Go to [Contents](#contents)