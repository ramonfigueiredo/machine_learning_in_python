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
	2. [Hierarchical Clustering](#hierarchical-clustering)
5. [Association Rule Learning](#association-rule-learning)
	1. [Apriori](#apriori)
6. [Reinforcement Learning](#reinforcement-learning)
	1. [Upper Confidence Bound](#upper-confidence-bound)
	2. [Thompson Sampling](#thompson-sampling)
7. [Natural Language Processing](#natural-language-processing)
8. [Deep Learning](#deep-learning)
	1. [Artificial Neural Networks](#artificial-neural-networks)
	2. [Convolutional Neural Networks](#convolutional-neural-networks)
9. [Dimensionality Reduction](#dimensionality-reduction)
	1. [Principal Component Analysis](#principal-component-analysis)
	2. [Linear Discriminant Analysis](#linear-discriminant-analysis)
	3. [Kernel PCA](#kernel-pca)
10. [Model Selection](#model-selection)
	1. [K-Fold Cross Validation](#k-fold-cross-validation)
	2. [Grid Search](#grid-search)
11. [Boosting](#boosting)
	1. [XG Boost](#xg-boost)
12. [Metrics using the Confusion Matrix](#metrics-using-the-confusion-matrix)
13. [How to run the Python program](#how-to-run-the-python-program)

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
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/1_simple_linear_regression/Visualising-the-Training-set-results.png)
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


| Classification Model 			  |	Pros 																					 	 | Cons |
| ------------------------------- |	-------------------------------------------------------------------------------------------- | ---- |
| Logistic Regression 			  | Probabilistic approach, gives informations about statistical significance of features    	 | The Logistic Regression Assumptions 	 |
| K-Nearest Neighbors (K-NN)	  | Simple to understand, fast and efficient 					 							 	 | Need to choose the number of neighbours k |
| Support Vector Machine (SVM) 	  | Performant, not biased by outliers, not sensitive to overfitting 		 				 	 | Not appropriate for non linear problems, not the best choice for large number of features |
| Kernel SVM  	  				  | High performance on nonlinear problems, not biased by outliers, not sensitive to overfitting | Not the best choice for large number of features, more complex|
| Naive Bayes 		  			  | Efficient, not biased by outliers, works on nonlinear problems, probabilistic approach 		 | Based on the assumption that features have same statistical relevance |
| Decision Tree Classification	  | Interpretability, no need for feature scaling, works on both linear / nonlinear problems 	 | Poor results on too small datasets, overfitting can easily occur |
| Random Forest Classification	  | Powerful and accurate, good performance on many problems, including non linear | No interpretability, overfitting can easily occur, need to choose the number of trees |

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

| Regression Model 				  |	Pros 																								   | Cons |
| ------------------------------- |	------------------------------------------------------------------------------------------------------------- | ---- |
| K-Means 			  			  | Simple to understand, easily adaptable, works well on small or large datasets, fast, efficient and performant | Need to choose the number of clusters |
| Hierarchical Clustering 		  | The optimal number of clusters can be obtained by the model itself, practical visualisation with the dendrogram | Not appropriate for large datasets |

### K-Means Clustering

a.  [kmeans.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/1_k_means_clustering/kmeans.py)

* Importing the dataset ([Mall_Customers.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/1_k_means_clustering/Mall_Customers.csv))
* Using the [Elbow method](https://en.wikipedia.org/wiki/Elbow_method_(clustering)) to find the optimal number of clusters
	* The Elbow method is a heuristic method of interpretation and validation of consistency within cluster analysis designed to help finding the appropriate number of clusters in a dataset
* Using [k-means++](https://en.wikipedia.org/wiki/K-means%2B%2B) to select initial cluster centers for k-mean clustering in a smart way to speed up convergence
* Plotting the Elbow method
	* The Elbow method uses the [Within-Cluster Sum of Squares (WCSS)](https://en.wikipedia.org/wiki/K-means_clustering) metric = Sum of squared distances of samples to their closest cluster center.
	* According to the Elbow method the best number of cluster in the mall customers dataset ([Mall_Customers.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/1_k_means_clustering/Mall_Customers.csv)) is 5
* Fitting K-Means to the dataset. The fit method returns for each observation which cluster it belongs to.

![The Elbow Method](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/1_k_means_clustering/The-Elbow-Method.png)

* Visualising the clusters
	* Cluster 1 has high income and low spending score. A better name for this cluster of clients as "Careful clients"
	* Cluster 2 has average income and average spending score. A better name for this cluster of clients as "Standard clients"
	* Cluster 3 has high income and high spending score. A better name for this cluster of clients as "Target clients"
		* Therefore, cluster 3 is the cluster of clients that would be the main potential target of the mall marketing campaigns and it would be very insighful for them all to understand what kind of products are bought by the clients in this cluster 
	* Cluster 4 has low income and high spending score. A better name for this cluster of clients as "Careless clients" 
	* Cluster 5 has low income and low spending score. A better name for this cluster of clients as "Sensible clients"
	
![Clusters of customers](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/1_k_means_clustering/Clusters-of-customers.png)

Go to [Contents](#contents)

### Hierarchical Clustering

a.  [hierarchical_clustering.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/2_hierarchical_clustering/hierarchical_clustering.py)

* Importing the dataset ([Mall_Customers.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/2_hierarchical_clustering/Mall_Customers.csv))
* Using the [dendrogram](https://en.wikipedia.org/wiki/Dendrogram) to find the optimal number of clusters
* Fitting Hierarchical Clustering to the dataset. The fit method returns for each observation which cluster it belongs to.
* Plotting the Dendrogram (euclidean distance and the ward linkage criterion)
	* According to the Dendrogram the best number of cluster in the mall customers dataset ([Mall_Customers.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/1_k_means_clustering/Mall_Customers.csv)) is 5

![Dendrogram](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/2_hierarchical_clustering/Dendrogram.png)

![Dendrogram with 5 clusters](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/2_hierarchical_clustering/Dendrogram-Largest-distance-5_clusters.png)

* Visualising the clusters
	* Cluster 1 has high income and low spending score. A better name for this cluster of clients as "Careful clients"
	* Cluster 2 has average income and average spending score. A better name for this cluster of clients as "Standard clients"
	* Cluster 3 has high income and high spending score. A better name for this cluster of clients as "Target clients"
		* Therefore, cluster 3 is the cluster of clients that would be the main potential target of the mall marketing campaigns and it would be very insighful for them all to understand what kind of products are bought by the clients in this cluster 
	* Cluster 4 has low income and high spending score. A better name for this cluster of clients as "Careless clients" 
	* Cluster 5 has low income and low spending score. A better name for this cluster of clients as "Sensible clients"
	
![Clusters of customers](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/2_hierarchical_clustering/Clusters-of-customers.png)

* Clusters visialization with different distance metrics and different linkage criterion
	* [See clusters of customers with **cosine** distance and 3 different linkage criterion (**average**, **complete**, and **single**)](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/2_hierarchical_clustering/Clusters-of-customers-cosine-distance)
	* [See clusters of customers with **euclidean** distance and 3 different linkage criterion (**ward**, **average**, **complete**, and **single**)](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/2_hierarchical_clustering/Clusters-of-customers-euclidean-distance)
	* [See clusters of customers with **L1** distance and 3 different linkage criterion (**average**, **complete**, and **single**)](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/2_hierarchical_clustering/Clusters-of-customers-L1-distance)
	* [See clusters of customers with **L2** distance and 3 different linkage criterion (**average**, **complete**, and **single**)](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/2_hierarchical_clustering/Clusters-of-customers-L2-distance)
	* [See clusters of customers with **manhattan** distance and 3 different linkage criterion (**average**, **complete**, and **single**)](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/4_clustering/2_hierarchical_clustering/Clusters-of-customers-manhattan-distance)

Go to [Contents](#contents)

## Association Rule Learning

### Apriori

a.  [apriori.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/5_association_rule_learning/1_apriori/apriori.py)

* Importing the dataset ([Market_Basket_Optimisation.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/5_association_rule_learning/1_apriori/Market_Basket_Optimisation.csv))
	* The dataset describes a store located in one of the most popular places in the south of France. So, a lot of people go into the store.
	* And therefore the manager of the store noticed and calculated that on average each customer goes and buys something to the store once a week.
	* This dataset contains 7500 transactions of all the different customers that bought a basket of products in a whole week.
	* Indeed the manage took it as the basis of its analysis because since each customer is going an average once a week to the store then the transaction registered over a week is quite representative of what customers want to buy.
	* So, based on all these 7500 transactions our machine learning model (*apriori*) is going to learn the different associations it can make to actually understand the rules.
	* Such as if customers buy this product then they're likely to buy this other set of products.
	* Each line in the database corresponds to a specific customer who bought a specific basket of product. 
	* For example, in line 2 the customer bought burgers, meatballs, and eggs.
* Creating list of transactions
* Training Apriori on the dataset
* Visualising the results

Go to [Contents](#contents)

## Reinforcement Learning

### Upper Confidence Bound

a.  [random_selection.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/random_selection.py)

* Importing the dataset ([Ads_CTR_Optimisation.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/Ads_CTR_Optimisation.csv))
* Implementing Random Selection
* Visualising the results

![Random selection - Histogram of ads selections](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/Random-selection_Histogram-of-ads-selections.png)

b.  [upper_confidence_bound.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/upper_confidence_bound.py)

* Importing the dataset ([Ads_CTR_Optimisation.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/Ads_CTR_Optimisation.csv))
* Implementing UCB
* Visualising the results

![Upper Confidence Bound (UCB) - Histogram of ads selections](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/UCB-Histogram-of-ads-selections.png)

#### UCB algorithm

**Step 1.** At each round n, we consider two numbers for each ad i:

* ![equation 1](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/equation1.gif) - the number of times the ad i was selected up to round n,

* ![equation 2](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/equation2.gif) - the sum of rewards of the ad i up to round n.

**Step 2.** From these two numbers we compute:

* the average reward of ad i up to round n

![equation 3](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/equation3.gif)

* the confidence interval ![equation 4](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/equation4.gif) at round n with

![equation 4](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/equation5.gif)
		
**Step 3.** We select the ad i that has the maximum UCB ![equation 6](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/equation6.gif)

Go to [Contents](#contents)

### Thompson Sampling

a.  [random_selection.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/2_thompson_sampling/random_selection.py)

* Importing the dataset ([Ads_CTR_Optimisation.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/2_thompson_sampling/Ads_CTR_Optimisation.csv))
* Implementing Random Selection
* Visualising the results

![Random selection - Histogram of ads selections](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/2_thompson_sampling/Random-selection_Histogram-of-ads-selections.png)

b.  [thompson_sampling.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/2_thompson_sampling/thompson_sampling.py)

* Importing the dataset ([Ads_CTR_Optimisation.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/2_thompson_sampling/Ads_CTR_Optimisation.csv))
* Implementing Thompson Sampling
* Visualising the results

![Thompson Sampling - Histogram of ads selections](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/2_thompson_sampling/Thompson_Sampling-Histogram-of-ads-selections.png)

#### Thompson Sampling algorithm

**Step 1.** At each round n, we consider two numbers for each ad i:

* ![equation 1](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/2_thompson_sampling/equation1.gif) - the number of times the ad i got reward 1 up to round n,

* ![equation 2](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/2_thompson_sampling/equation2.gif) - the number of times the ad i got reward 0 up to round n.

**Step 2.** For each ad i, we take a random draw from the distribution below:

![equation 3](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/2_thompson_sampling/equation3.gif)

**Step 3.** We select the ad that has the highest ![equation 4](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/6_reinforcement_learning/2_thompson_sampling/equation4.gif)

Go to [Contents](#contents)

## Natural Language Processing

a.  [natural_language_processing.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/7_natural_language_processing/natural_language_processing.py)

* Importing the dataset ([Restaurant_Reviews.tsv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/7_natural_language_processing/Restaurant_Reviews.tsv))
* Cleaning the texts (remove text different of a-z or A-Z removing stop words, suffix stripping using Porter Stemming Algorithm)
* Creating the Bag of Words model
* Splitting the dataset into the Training set and Test set
* Fitting Naive Bayes to the Training set
* Predicting the Test set results
* Making the Confusion Matrix
* Calculating metrics using the confusion matrix

See [Metrics using the Confusion Matrix](#metrics-using-the-confusion-matrix)

### Algorithm output

```
Predicting the Test set results
 [1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 0 0 1 1 1 0 1 1 1 0 1 1 1 1 1 0 1
 0 1 1 1 1 1 0 0 0 1 1 0 0 1 1 1 1 1 0 1 1 0 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1
 0 1 1 0 0 1 0 1 1 0 1 1 1 0 1 1 0 1 0 0 1 1 1 1 1 1 0 1 1 1 0 1 1 1 0 0 0
 1 0 1 1 0 1 1 1 1 1 0 1 1 0 0 1 1 0 1 1 1 0 0 1 1 1 1 1 1 0 1 1 0 1 0 1 1
 1 1 1 0 1 1 1 0 1 1 1 1 1 0 0 1 0 0 1 0 0 0 0 1 1 0 0 1 0 1 0 0 1 0 0 1 0
 1 0 1 0 1 1 0 1 1 1 0 1 1 1 1]


Confusion Matrix
 [[55 42]
 [12 91]]


True Positive (TP): 55
False Negative (FN): 42
True Negative (TN): 12
False Positive (FP): 91


Accuracy = (TP + TN) / (TP + TN + FP + FN): 33.50 %
Recall = TP / (TP + FN): 56.70 %
Precision = TP / (TP + FP): 37.67 %
Fmeasure = (2 * recall * precision) / (recall + precision): 45.27 %
```

Go to [Contents](#contents)

## Deep Learning

### Artificial Neural Networks

a.  [ann.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/8_deep_learning/1_artificial_neural_networks/ann.py)

* Importing the dataset ([Churn_Modelling.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/8_deep_learning/1_artificial_neural_networks/Churn_Modelling.csv))
* Encoding categorical data
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Creating the Artificial Neural Networks (ANN) using [Keras](https://keras.io/)
	* Initialising the ANN
	* Adding the input layer and the first hidden layer
	* Adding the second hidden layer
	* Adding the output layer
	* Compiling the ANN
	* Fitting the ANN to the Training set
* Predicting the Test set results
* Creating the Confusion Matrix
* Calculating metrics using the confusion matrix

#### Training the ANN with Stochastic Gradient Descent

**Step 1.** Randomly initialise the weights to small numbers close to 0 (but not 0).

**Step 2.** Input the first observation of your dataset in the input layer, each feature in one input node.

**Step 3.** Forward-Propagation: from left to right, the neurons are activated in a way that the impact of each neuron's activation is limited by the weights. Propagate the activations until getting the predicted results y.

**Step 4.** Compare the predicted results to the actual result. Measure the generated error.

**Step 5.** Back-Propagation: fron right to left, the error is back-propagated. Update the weights according to how much they are responsible for the error. The learning rate decides by how much we update the weights.

**Step 6.** Repeat Steps 1 to 5 and update the weights after each observation (Reinforcement Learning). Or: Repeat Steps 1 to 5 but update the weights only after a batch of observation (Batch Learning).

**Step 7.** When the whole training set passed through the ANN, that makes an epoch. Redo more epochs.

See [Metrics using the Confusion Matrix](#metrics-using-the-confusion-matrix)

#### ANN algorithm output using Keras and TensorFlow (CPU)

##### Computer settings

* Mac OS Mojave (version 10.14.6)
* MacBook Pro (15-inch, 2017)
* Processor 2.8 GHz Intel Core i7
* Memory 16 GB 2133 MHz LPDDR3

```
Using TensorFlow backend.

Epoch 1/100
8000/8000 [==============================] - 1s 102us/step - loss: 0.4960 - accuracy: 0.7943
Epoch 2/100
8000/8000 [==============================] - 1s 82us/step - loss: 0.4288 - accuracy: 0.7960
Epoch 3/100
8000/8000 [==============================] - 1s 78us/step - loss: 0.4237 - accuracy: 0.7960
Epoch 4/100
8000/8000 [==============================] - 1s 78us/step - loss: 0.4201 - accuracy: 0.8076
Epoch 5/100
8000/8000 [==============================] - 1s 79us/step - loss: 0.4175 - accuracy: 0.8224
Epoch 6/100
8000/8000 [==============================] - 1s 78us/step - loss: 0.4154 - accuracy: 0.8269
Epoch 7/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4142 - accuracy: 0.8290
Epoch 8/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4125 - accuracy: 0.8295
Epoch 9/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4120 - accuracy: 0.8311
Epoch 10/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4107 - accuracy: 0.8331
Epoch 11/100
8000/8000 [==============================] - 1s 76us/step - loss: 0.4101 - accuracy: 0.8320
Epoch 12/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4092 - accuracy: 0.8332
Epoch 13/100
8000/8000 [==============================] - 1s 76us/step - loss: 0.4085 - accuracy: 0.8354
Epoch 14/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4083 - accuracy: 0.8328
Epoch 15/100
8000/8000 [==============================] - 1s 79us/step - loss: 0.4074 - accuracy: 0.8351
Epoch 16/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4071 - accuracy: 0.8351
Epoch 17/100
8000/8000 [==============================] - 1s 76us/step - loss: 0.4066 - accuracy: 0.8344
Epoch 18/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4063 - accuracy: 0.8336
Epoch 19/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4057 - accuracy: 0.8342
Epoch 20/100
8000/8000 [==============================] - 1s 78us/step - loss: 0.4059 - accuracy: 0.8324
Epoch 21/100
8000/8000 [==============================] - 1s 80us/step - loss: 0.4048 - accuracy: 0.8353
Epoch 22/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4051 - accuracy: 0.8342
Epoch 23/100
8000/8000 [==============================] - 1s 76us/step - loss: 0.4052 - accuracy: 0.8344
Epoch 24/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4045 - accuracy: 0.8354
Epoch 25/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4041 - accuracy: 0.8354
Epoch 26/100
8000/8000 [==============================] - 1s 78us/step - loss: 0.4044 - accuracy: 0.8342
Epoch 27/100
8000/8000 [==============================] - 1s 78us/step - loss: 0.4044 - accuracy: 0.8351
Epoch 28/100
8000/8000 [==============================] - 1s 80us/step - loss: 0.4037 - accuracy: 0.8341
Epoch 29/100
8000/8000 [==============================] - 1s 82us/step - loss: 0.4034 - accuracy: 0.8346
Epoch 30/100
8000/8000 [==============================] - 1s 79us/step - loss: 0.4035 - accuracy: 0.8354
Epoch 31/100
8000/8000 [==============================] - 1s 79us/step - loss: 0.4030 - accuracy: 0.8335
Epoch 32/100
8000/8000 [==============================] - 1s 80us/step - loss: 0.4033 - accuracy: 0.8342
Epoch 33/100
8000/8000 [==============================] - 1s 79us/step - loss: 0.4031 - accuracy: 0.8344
Epoch 34/100
8000/8000 [==============================] - 1s 78us/step - loss: 0.4034 - accuracy: 0.8341
Epoch 35/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4030 - accuracy: 0.8346
Epoch 36/100
8000/8000 [==============================] - 1s 78us/step - loss: 0.4025 - accuracy: 0.8346
Epoch 37/100
8000/8000 [==============================] - 1s 78us/step - loss: 0.4028 - accuracy: 0.8334
Epoch 38/100
8000/8000 [==============================] - 1s 79us/step - loss: 0.4026 - accuracy: 0.8350
Epoch 39/100
8000/8000 [==============================] - 1s 79us/step - loss: 0.4028 - accuracy: 0.8338
Epoch 40/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4025 - accuracy: 0.8350
Epoch 41/100
8000/8000 [==============================] - 1s 81us/step - loss: 0.4021 - accuracy: 0.8332
Epoch 42/100
8000/8000 [==============================] - 1s 78us/step - loss: 0.4024 - accuracy: 0.8356
Epoch 43/100
8000/8000 [==============================] - 1s 76us/step - loss: 0.4023 - accuracy: 0.8339
Epoch 44/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4019 - accuracy: 0.8339
Epoch 45/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4022 - accuracy: 0.8353
Epoch 46/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4019 - accuracy: 0.8328
Epoch 47/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4023 - accuracy: 0.8345
Epoch 48/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4020 - accuracy: 0.8339
Epoch 49/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4021 - accuracy: 0.8354
Epoch 50/100
8000/8000 [==============================] - 1s 76us/step - loss: 0.4020 - accuracy: 0.8334
Epoch 51/100
8000/8000 [==============================] - 1s 75us/step - loss: 0.4020 - accuracy: 0.8345
Epoch 52/100
8000/8000 [==============================] - 1s 78us/step - loss: 0.4017 - accuracy: 0.8342
Epoch 53/100
8000/8000 [==============================] - 1s 75us/step - loss: 0.4021 - accuracy: 0.8340
Epoch 54/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4018 - accuracy: 0.8353
Epoch 55/100
8000/8000 [==============================] - 1s 76us/step - loss: 0.4016 - accuracy: 0.8339
Epoch 56/100
8000/8000 [==============================] - 1s 78us/step - loss: 0.4017 - accuracy: 0.8345
Epoch 57/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4020 - accuracy: 0.8338
Epoch 58/100
8000/8000 [==============================] - 1s 76us/step - loss: 0.4018 - accuracy: 0.8335
Epoch 59/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4018 - accuracy: 0.8353
Epoch 60/100
8000/8000 [==============================] - 1s 76us/step - loss: 0.4018 - accuracy: 0.8336
Epoch 61/100
8000/8000 [==============================] - 1s 78us/step - loss: 0.4017 - accuracy: 0.8339
Epoch 62/100
8000/8000 [==============================] - 1s 79us/step - loss: 0.4017 - accuracy: 0.8341
Epoch 63/100
8000/8000 [==============================] - 1s 79us/step - loss: 0.4018 - accuracy: 0.8340
Epoch 64/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4018 - accuracy: 0.8339
Epoch 65/100
8000/8000 [==============================] - 1s 79us/step - loss: 0.4016 - accuracy: 0.8355
Epoch 66/100
8000/8000 [==============================] - 1s 79us/step - loss: 0.4020 - accuracy: 0.8341
Epoch 67/100
8000/8000 [==============================] - 1s 81us/step - loss: 0.4018 - accuracy: 0.8347
Epoch 68/100
8000/8000 [==============================] - 1s 78us/step - loss: 0.4013 - accuracy: 0.8340
Epoch 69/100
8000/8000 [==============================] - 1s 79us/step - loss: 0.4016 - accuracy: 0.8346
Epoch 70/100
8000/8000 [==============================] - 1s 78us/step - loss: 0.4020 - accuracy: 0.8342
Epoch 71/100
8000/8000 [==============================] - 1s 79us/step - loss: 0.4015 - accuracy: 0.8346
Epoch 72/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4016 - accuracy: 0.8339
Epoch 73/100
8000/8000 [==============================] - 1s 78us/step - loss: 0.4012 - accuracy: 0.8350
Epoch 74/100
8000/8000 [==============================] - 1s 81us/step - loss: 0.4015 - accuracy: 0.8335
Epoch 75/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4014 - accuracy: 0.8340
Epoch 76/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4013 - accuracy: 0.8338
Epoch 77/100
8000/8000 [==============================] - 1s 78us/step - loss: 0.4016 - accuracy: 0.8344
Epoch 78/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4015 - accuracy: 0.8347
Epoch 79/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4015 - accuracy: 0.8331
Epoch 80/100
8000/8000 [==============================] - 1s 76us/step - loss: 0.4015 - accuracy: 0.8341
Epoch 81/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4015 - accuracy: 0.8349
Epoch 82/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4018 - accuracy: 0.8341
Epoch 83/100
8000/8000 [==============================] - 1s 76us/step - loss: 0.4013 - accuracy: 0.8339
Epoch 84/100
8000/8000 [==============================] - 1s 76us/step - loss: 0.4011 - accuracy: 0.8349
Epoch 85/100
8000/8000 [==============================] - 1s 76us/step - loss: 0.4017 - accuracy: 0.8346
Epoch 86/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4014 - accuracy: 0.8338
Epoch 87/100
8000/8000 [==============================] - 1s 78us/step - loss: 0.4016 - accuracy: 0.8349
Epoch 88/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4014 - accuracy: 0.8353
Epoch 89/100
8000/8000 [==============================] - 1s 76us/step - loss: 0.4013 - accuracy: 0.8329
Epoch 90/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4016 - accuracy: 0.8339
Epoch 91/100
8000/8000 [==============================] - 1s 79us/step - loss: 0.4013 - accuracy: 0.8350
Epoch 92/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4014 - accuracy: 0.8340
Epoch 93/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4014 - accuracy: 0.8334
Epoch 94/100
8000/8000 [==============================] - 1s 76us/step - loss: 0.4016 - accuracy: 0.8329
Epoch 95/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4015 - accuracy: 0.8354
Epoch 96/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4013 - accuracy: 0.8349
Epoch 97/100
8000/8000 [==============================] - 1s 77us/step - loss: 0.4013 - accuracy: 0.8336
Epoch 98/100
8000/8000 [==============================] - 1s 76us/step - loss: 0.4014 - accuracy: 0.8329
Epoch 99/100
8000/8000 [==============================] - 1s 76us/step - loss: 0.4014 - accuracy: 0.8339
Epoch 100/100
8000/8000 [==============================] - 1s 76us/step - loss: 0.4011 - accuracy: 0.8345

Predicting the Test set results
 [0 1 0 ... 0 0 0]


Confusion Matrix
 [[1547   48]
 [ 266  139]]


True Positive (TP): 1547
False Negative (FN): 48
True Negative (TN): 266
False Positive (FP): 139


Accuracy = (TP + TN) / (TP + TN + FP + FN): 90.65 %
Recall = TP / (TP + FN): 96.99 %
Precision = TP / (TP + FP): 91.76 %
Fmeasure = (2 * recall * precision) / (recall + precision): 94.30 %

```

Go to [Contents](#contents)

### Convolutional Neural Networks

a.  [cnn.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/8_deep_learning/2_convolutional_neural_networks/cnn.py)

* Using a dataset with 10000 images of cats and dogs ([cats and dogs dataset](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/8_deep_learning/2_convolutional_neural_networks/dataset))
	* Training set: 8000 (4000 cat images + 4000 dogs images)
	* Test set: 2000 (1000 cat images + 1000 dogs images)
* Creating the Convolutional Neural Network using [Keras](https://keras.io/)
	* Initialising the CNN
	* Convolution
	* Pooling
	* Adding a second convolutional layer
	* Flattening
	* Full connection
	* Compiling the CNN
	* Fitting the CNN to the images

#### Training the CNN

**Step 1.** Convolution

**Step 2.** Max Pooling

**Step 3.** Flattening

**Step 4.** Full connection

See [Metrics using the Confusion Matrix](#metrics-using-the-confusion-matrix)

#### CNN algorithm output using Keras and TensorFlow (CPU)

* **Note:** 
	* I executed  this code using tensorflow (CPU). Execute this code using CPU takes lots of time. 
	* If you have GPU you can use tensorflow-gpu. 
	* The following GPU-enabled devices are supported: NVIDIA(R) GPU card with CUDA(R) Compute Capability 3.5 or higher. See the list of [CUDA-enabled GPU cards](https://developer.nvidia.com/cuda-gpus).
	* The following NVIDIA(R) software must be installed on your system:
		* [NVIDIA(R) GPU drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us) — CUDA 10.0 requires 410.x or higher.
		* [CUDA(R) Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) — TensorFlow supports CUDA 10.0 (TensorFlow >= 1.13.0)
		* [CUPTI](https://docs.nvidia.com/cuda/cupti/) ships with the CUDA Toolkit.
		* [cuDNN SDK](https://developer.nvidia.com/cudnn) (>= 7.4.1)
		* (Optional) [TensorRT 5.0](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html) to improve latency and throughput for inference on some models.

##### Computer settings

* Windows 10 Professional (x64)
* Processor Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz
* Memory 32 GB

```
Using TensorFlow backend.

Found 8000 images belonging to 2 classes.
Found 2000 images belonging to 2 classes.
Epoch 1/25
8000/8000 [==============================] - 887s 111ms/step - loss: 0.3579 - accuracy: 0.8324 - val_loss: 0.5902 - val_accuracy: 0.8086
Epoch 2/25
8000/8000 [==============================] - 759s 95ms/step - loss: 0.1079 - accuracy: 0.9588 - val_loss: 0.5788 - val_accuracy: 0.8000
Epoch 3/25
8000/8000 [==============================] - 716s 90ms/step - loss: 0.0539 - accuracy: 0.9813 - val_loss: 0.6424 - val_accuracy: 0.7990
Epoch 4/25
8000/8000 [==============================] - 717s 90ms/step - loss: 0.0386 - accuracy: 0.9867 - val_loss: 0.9461 - val_accuracy: 0.8047
Epoch 5/25
8000/8000 [==============================] - 713s 89ms/step - loss: 0.0308 - accuracy: 0.9896 - val_loss: 1.3553 - val_accuracy: 0.7848
Epoch 6/25
8000/8000 [==============================] - 713s 89ms/step - loss: 0.0259 - accuracy: 0.9913 - val_loss: 1.3581 - val_accuracy: 0.7889
Epoch 7/25
8000/8000 [==============================] - 714s 89ms/step - loss: 0.0224 - accuracy: 0.9927 - val_loss: 0.9129 - val_accuracy: 0.8069
Epoch 8/25
8000/8000 [==============================] - 726s 91ms/step - loss: 0.0189 - accuracy: 0.9939 - val_loss: 1.2980 - val_accuracy: 0.7935
Epoch 9/25
8000/8000 [==============================] - 736s 92ms/step - loss: 0.0178 - accuracy: 0.9943 - val_loss: 1.9009 - val_accuracy: 0.7885
Epoch 10/25
8000/8000 [==============================] - 716s 90ms/step - loss: 0.0163 - accuracy: 0.9949 - val_loss: 1.4097 - val_accuracy: 0.7889
Epoch 11/25
8000/8000 [==============================] - 713s 89ms/step - loss: 0.0138 - accuracy: 0.9957 - val_loss: 0.7039 - val_accuracy: 0.7976
Epoch 12/25
8000/8000 [==============================] - 716s 89ms/step - loss: 0.0130 - accuracy: 0.9959 - val_loss: 1.4262 - val_accuracy: 0.7914
Epoch 13/25
8000/8000 [==============================] - 712s 89ms/step - loss: 0.0123 - accuracy: 0.9963 - val_loss: 0.7608 - val_accuracy: 0.7976
Epoch 14/25
8000/8000 [==============================] - 714s 89ms/step - loss: 0.0122 - accuracy: 0.9963 - val_loss: 2.7076 - val_accuracy: 0.8005
Epoch 15/25
8000/8000 [==============================] - 715s 89ms/step - loss: 0.0108 - accuracy: 0.9967 - val_loss: 3.5931 - val_accuracy: 0.7930
Epoch 16/25
8000/8000 [==============================] - 713s 89ms/step - loss: 0.0104 - accuracy: 0.9969 - val_loss: 0.6374 - val_accuracy: 0.7970
Epoch 17/25
8000/8000 [==============================] - 712s 89ms/step - loss: 0.0100 - accuracy: 0.9969 - val_loss: 1.3442 - val_accuracy: 0.8016
Epoch 18/25
8000/8000 [==============================] - 712s 89ms/step - loss: 0.0091 - accuracy: 0.9973 - val_loss: 2.7414 - val_accuracy: 0.8020
Epoch 19/25
8000/8000 [==============================] - 714s 89ms/step - loss: 0.0091 - accuracy: 0.9973 - val_loss: 1.3481 - val_accuracy: 0.7944
Epoch 20/25
8000/8000 [==============================] - 712s 89ms/step - loss: 0.0089 - accuracy: 0.9974 - val_loss: 4.1220 - val_accuracy: 0.7976
Epoch 21/25
8000/8000 [==============================] - 714s 89ms/step - loss: 0.0086 - accuracy: 0.9975 - val_loss: 0.8613 - val_accuracy: 0.7923
Epoch 22/25
8000/8000 [==============================] - 716s 90ms/step - loss: 0.0088 - accuracy: 0.9976 - val_loss: 3.9867 - val_accuracy: 0.7960
Epoch 23/25
8000/8000 [==============================] - 731s 91ms/step - loss: 0.0076 - accuracy: 0.9977 - val_loss: 1.3609 - val_accuracy: 0.7892
Epoch 24/25
8000/8000 [==============================] - 749s 94ms/step - loss: 0.0074 - accuracy: 0.9978 - val_loss: 2.1906 - val_accuracy: 0.7942
Epoch 25/25
8000/8000 [==============================] - 718s 90ms/step - loss: 0.0067 - accuracy: 0.9979 - val_loss: 1.2555 - val_accuracy: 0.8042
```

Go to [Contents](#contents)

## Dimensionality Reduction

### Principal Component Analysis

The goal of Principal Component Analysis (PCA) is identify patterns in data and detect the correlation between variables.

PCA can be used to reduce the dimensions of a d-dimensional dataset by projecting it onto a (k)-dimensional subspace (where k < d).

PCA is an unsupervised learning algorithm and a linear transformation technique.

a. [pca.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/9_dimensionality_reduction/1_principal_component_analysis/pca.py)

* Importing the dataset ([Wine.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/9_dimensionality_reduction/1_principal_component_analysis/Wine.csv))
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Applying Principal Component Analysis (PCA)
* Fitting Logistic Regression to the Training set
* Predicting the Test set results
* Creating the Confusion Matrix
* Visualising the Training and Test set results
* Calculating metrics using the confusion matrix

* Visualising the Training set results
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/9_dimensionality_reduction/1_principal_component_analysis/Visualising-the-Training-set-results.png)
* Visualising the Test set results
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/9_dimensionality_reduction/1_principal_component_analysis/Visualising-the-Test-set-results.png)

#### PCA algorithm

Step 1: Standardize the data.

Step 2: Obtain the Eigenvectors and Eigenvalues from the covariance matrix or correlation matrix, or perform Singular Vector Decomposition.

Step 3: Sort eigenvalues in descending order and choose the *k* eigenvectors that correspond to the k largest eigenvalues where k is the number of dimensions of the new feature subspace (*k <= d*).

Step 4: Construct the projection matrix **W** from the selected *k* eigenvalues.

Step 5: Transform the original dataset **X** via **W** to obtain a *k*-dimensional feature subspace **Y**.

PCA: [https://plot.ly/ipython-notebooks/principal-component-analysis/](https://plot.ly/ipython-notebooks/principal-component-analysis/)

#### PCA algorithm output

See [Metrics using the Confusion Matrix](#metrics-using-the-confusion-matrix)

```
Predicting the Test set results
 [1 3 2 1 2 1 1 3 2 2 3 3 1 2 3 2 1 1 2 1 2 1 1 2 2 2 2 2 2 3 1 1 2 1 1 1]

Confusion Matrix
 [[14  0  0]
 [ 1 15  0]
 [ 0  0  6]]


Classification report
               precision    recall  f1-score   support

           1      0.933     1.000     0.966        14
           2      1.000     0.938     0.968        16
           3      1.000     1.000     1.000         6

    accuracy                          0.972        36
   macro avg      0.978     0.979     0.978        36
weighted avg      0.974     0.972     0.972        36



True Positive (TP) of class 1: 14
True Positive (TP) of class 2: 15
True Positive (TP) of class 3: 6

ACCURACY, PRECISION, RECALL, F1-SCORE FOR CLASS 1

Accuracy (class 1) = TP (class 1) + cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2] / sum_matrix_values: 97.22 %

Precision (class 1) = TP (class 1) / (cm[0][0] + cm[1][0] + cm[2][0]): 93.33 %

Recall (class 1) = TP (class 1) / (cm[0][0] + cm[0][1] + cm[0][2]): 100.00 %

F1-Score (class 1) = (2 * recall_class1 * precision_class1) / (recall_class1 + precision_class1): 96.55 %

ACCURACY, PRECISION, RECALL, F1-SCORE FOR CLASS 2

Accuracy (class 2) = TP (class 2) + cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2] / sum_matrix_values: 97.22 %

Precision (class 2) = TP (class 2) / (cm[0][1] + cm[1][1] + cm[2][1]): 100.00 %

Recall (class 2) = TP (class 2) / (cm[1][0] + cm[1][1] + cm[1][2]): 93.75 %

F1-Score (class 2) = (2 * recall_class2 * precision_class2) / (recall_class2 + precision_class2): 96.77 %

PRECISION, RECALL, F1-SCORE FOR CLASS 3

Accuracy (class 3) = TP (class 3) + cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1] / sum_matrix_values: 100.00 %

Precision (class 3) = TP (class 3) / (cm[0][2] + cm[1][2] + cm[2][2]): 100.00 %

Recall (class 3) = TP (class 3) / (cm[2][0] + cm[2][1] + cm[2][2]): 100.00 %

F1-Score (class 3) = (2 * recall_class3 * precision_class3) / (recall_class3 + precision_class3): 100.00 %
```

Go to [Contents](#contents)

### Linear Discriminant Analysis

Linear Discriminant Analysis (LDA) is used as a dimensionality reduction technique and in the pre-processing step for pattern classification.

LDA has the goal to project a dataset onto a lower-dimensional space.

LDA differs from PCA because in addition to finding the component axises with LDA we are interested in the axes that maximize the separation between multiple classes.

In summary, LDA is to project a feature space (a dataset n-dimensional samples) onto a small subspace k (where K <= n - 1) while maintaining the class-discriminatory information. LDA is a supervised learning algorithm.

Both PCA and LDA are linear transformation techniques used for dimensional reduction. PCA is described as unsupervised but LDA is supervised because of the relation to the dependent variable.

a. [lda.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/9_dimensionality_reduction/2_linear_discriminant_analysis/lda.py)

* Importing the dataset ([Wine.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/9_dimensionality_reduction/2_linear_discriminant_analysis/Wine.csv))
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Applying Linear Discriminant Analysis (LDA)
* Fitting Logistic Regression to the Training set
* Predicting the Test set results
* Creating the Confusion Matrix

* Visualising the Training set results
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/9_dimensionality_reduction/2_linear_discriminant_analysis/Visualising-the-Training-set-results.png)
* Visualising the Test set results
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/9_dimensionality_reduction/2_linear_discriminant_analysis/Visualising-the-Test-set-results.png)

#### LDA algorithm

Step 1: Compute the *d*-dimensional mean vectors for the different classes from the dataset.

Step 2: Compute the scatter matrices (in-between-class and within-class scatter matrix).

Step 3: Compute the eigenvectors (**e1**,**e2**,...) and corresponding eigenvalues (*λ1*,*λ2*,...) for the scatter matrices.

Step 4: Sort the eigenvectors by decreasing eigenvalues and choose *k* eigenvectors with the largest eigenvalues to form a *d x k* dimensional matrix **W** (where every column represents an eigenvector).

Step 5: Use this *d x k* eigenvector matrix to transform the samples onto the new subspace. This can be summarized by the matrix multiplication: **Y = X x W** (where *X* is a *n x d*-dimensional matrix representing the *n* samples, and **y** are the transformed *n x k*-dimensional samples in the new subspace).

LDA: [https://plot.ly/ipython-notebooks/principal-component-analysis/](https://sebastianraschka.com/Articles/2014_python_lda.html)

![PCA and LDA](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/9_dimensionality_reduction/2_linear_discriminant_analysis/pca_lda.png)

Go to [Contents](#contents)

### Kernel PCA

Kernel PCA is a feature extraction technique adapted for non-linear problems where the data is not linearly separable.

Kernel PCA is a kernelizable version of PCA where we match the data to a higher dimension using the kernel trick, and then from there we extract sme new principal components. 

a. [kernel_pca.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/9_dimensionality_reduction/3_kernel_pca/kernel_pca.py)

* Importing the dataset ([Social_Network_Ads.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/9_dimensionality_reduction/3_kernel_pca/Social_Network_Ads.csv))
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Applying Kernel PCA
* Fitting Logistic Regression to the Training set
* Predicting the Test set results
* Creating the Confusion Matrix

* Visualising the Training set results
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/9_dimensionality_reduction/3_kernel_pca/Visualising-the-Training-set-results.png)
* Visualising the Test set results
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/9_dimensionality_reduction/3_kernel_pca/Visualising-the-Test-set-results.png)

#### Kernel PCA algorithm

The Gaussian RBF kernel is the must commonly used in Kernal PCA.

![Gaussian RBF kernel](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/9_dimensionality_reduction/3_kernel_pca/gaussian_rbf_kernel.png)

![Kernel PCA](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/9_dimensionality_reduction/3_kernel_pca/kernel_pca.png)

Go to [Contents](#contents)

## Model Selection

Model selection is the process of choosing between different machine learning approaches - e.g. SVM, logistic regression, etc - or choosing between different hyperparameters or sets of features for the same machine learning approach - e.g. deciding between the polynomial degrees/complexities for linear regression.

Every time we build a machine learning, we have two types of parameters: The first type is the parameters that the model learns (parameters that will change and find the optimal values by running the model) and the second type of parameters are the parameters that we choose ourselves. For example, the kernel parameter in the kernel as we model and these parameters are called the hyperparameters. So there are still places to improve the model because we can still choose optimal values for these parameters. And we can find these parameters using the Grid Search technique.

### K-Fold Cross Validation

So far, to evaluate the machine learning models we train the model in the training set and test its performance on the test set. That's a correct way of evaluating the model performance but is not the best one because we have a variance problem. The variance problem can be explained by the fact that when we get the accuracy on the test set, its performance can change on another test set. Judging our model performance only on one accuracy on one test set is not the most relevant way to evaluate the model.

Therefore, the best approach is to use the K-fold cross validation to fix the variance problem. It will fix this problem by splitting the dataset into 10 folds (most of the time k = 10) and we train our model on the last remaining folds. If the 10 folds we can make 10 different combinations of nine folds to train and 1 fold to test, that's means we can have 10 combinations of training sets and test sets. This will give a better idea of how the model performs because we can have an average of different accuracies up to ten evaluations and also compute the standard deviation to have a look in the variance. 

So eventually our analysis will be much more relevant and besides, we will know in which of the following bias-variance our dataset will be.

![Bias-variance tradeoff](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/10_model_selection_and_boosting/1_model_selection/bias-variance_tradeoff.png)

a. [k_fold_cross_validation.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/10_model_selection_and_boosting/1_model_selection/k_fold_cross_validation.py)

* Importing the dataset ([Social_Network_Ads.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/10_model_selection_and_boosting/1_model_selection/Social_Network_Ads.csv))
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Fitting Kernel SVM to the Training set
* Predicting the Test set results
* Creating the Confusion Matrix
* Applying k-Fold Cross Validation (K = 10)
* Accuracy in each of the 10 folds
* Average accuracy after 10-Fold Cross Validation
* Average standard deviation after 10-Fold Cross Validation

* Visualising the Training set results
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/10_model_selection_and_boosting/1_model_selection/10_fold_cross_validation-Visualising-the-Training-set-results.png)
* Visualising the Test set results
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/10_model_selection_and_boosting/1_model_selection/10_fold_cross_validation-Visualising-the-Test-set-results.png)

#### K-Fold Cross Validation steps

Step 1: Shuffle the dataset randomly.

Step 2: Split the dataset into k groups

Step 3: For each unique group:
* Take the group as a hold out or test data set
* Take the remaining groups as a training data set
* Fit a model on the training set and evaluate it on the test set
* Retain the evaluation score and discard the model

Step 4: Summarize the skill of the model using the sample of model evaluation scores

![K-Fold Cross Validation](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/10_model_selection_and_boosting/1_model_selection/k-fold_cross_validation.png)

Go to [Contents](#contents)

### Grid Search

The traditional way of performing hyperparameter optimization is to use grid search, or a parameter sweep, which is simply an exhaustive searching through a manually specified subset of the hyperparameter space of a learning algorithm. A grid search algorithm must be guided by some performance metric, typically measured by cross-validation on the training set or evaluation on a held-out validation set.

Since the parameter space of a machine learning algorithm may include real-valued or unbounded value spaces for certain parameters, manually set bounds, and discretization may be necessary before applying a grid search.

For example, a typical soft-margin SVM classifier equipped with an RBF kernel has at least two hyperparameters that need to be tuned for good performance on unseen data: a regularization constant C and a kernel hyperparameter K. Both parameters are continuous, so to perform grid search, one selects a finite set of "reasonable" values for each.

Grid search then trains an SVM with each pair (C, K) in the Cartesian product of these two sets and evaluates their performance on a held-out validation set (or by internal cross-validation on the training set, in which case multiple SVMs are trained per pair). Finally, the grid search algorithm outputs the settings that achieved the highest score in the validation procedure.

Grid search suffers from the curse of dimensionality but is often embarrassingly parallel because the hyperparameter settings it evaluates are typically independent of each other.

In summary, Grid Search is used to know which parameter to select when you choose a machine learning model and what is the optimal value of these hyperparameters.

a. [grid_search.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/10_model_selection_and_boosting/1_model_selection/grid_search.py)

* Importing the dataset ([Social_Network_Ads.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/10_model_selection_and_boosting/1_model_selection/Social_Network_Ads.csv))
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Fitting Kernel SVM to the Training set
* Predicting the Test set results
* Creating the Confusion Matrix
* Applying k-Fold Cross Validation (K = 10)
* Accuracy in each of the 10 folds
* Average accuracy after 10-Fold Cross Validation
* Average standard deviation after 10-Fold Cross Validation
* Grid Search: Best accuracy
* Grid Search: Best parameters

* Visualising the Training set results
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/10_model_selection_and_boosting/1_model_selection/grid_search-Visualising-the-Training-set-results.png)
* Visualising the Test set results
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/10_model_selection_and_boosting/1_model_selection/grid_search-Visualising-the-Test-set-results.png)

Go to [Contents](#contents)

## Boosting

### XG Boost

a. [xg_boost.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/10_model_selection_and_boosting/2_xg_boost/xg_boost.py)

* Importing the dataset ([Churn_Modelling.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/10_model_selection_and_boosting/2_xg_boost/Churn_Modelling.csv))
* Encoding categorical data
* Splitting the dataset into the Training set and Test set
* Fitting XGBoost to the Training set
* Predicting the Test set results
* Creating the Confusion Matrix
* Applying k-Fold Cross Validation (K = 10)
* Accuracy in each of the 10 folds
* Average accuracy after 10-Fold Cross Validation
* Average standard deviation after 10-Fold Cross Validation

Go to [Contents](#contents)

## Metrics using the Confusion Matrix 

### Confusion Matrix (Binary Classification)

![Confusion Matrix: Binary Classification](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/confusion_matrix-binary_classification.png)

### True Positive (TP), False Negative (FN), True Negative (TN), False Positive (FP)

* **True Positive (TP):** Observation is positive, and is predicted to be positive.
* **False Negative (FN):** Observation is positive, but is predicted negative.
* **True Negative (TN):** Observation is negative, and is predicted to be negative.
* **False Positive (FP):** Observation is negative, but is predicted positive.

### Classification Rate / Accuracy

Classification Rate or Accuracy is given by the relation:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

However, there are problems with accuracy.  It assumes equal costs for both kinds of errors. A 99% accuracy can be excellent, good, mediocre, poor or terrible depending upon the problem.

### Recall

Recall can be defined as the ratio of the total number of correctly classified positive examples divide to the total number of positive examples. High Recall indicates the class is correctly recognized (small number of FN).

Recall is given by the relation:

Recall = TP / (TP + FN)

### Precision

To get the value of precision we divide the total number of correctly classified positive examples by the total number of predicted positive examples. High Precision indicates an example labeled as positive is indeed positive (small number of FP).

Precision is given by the relation:

Precision = TP / (TP + FP)

High recall, low precision: 
This means that most of the positive examples are correctly recognized (low FN) but there are a lot of false positives.

Low recall, high precision: 
This shows that we miss a lot of positive examples (high FN) but those we predict as positive are indeed positive (low FP)

### F1-Score

Since we have two measures (Precision and Recall) it helps to have a measurement that represents both of them. We calculate an F1-Score (F-measure) which uses Harmonic Mean in place of Arithmetic Mean as it punishes the extreme values more.

The F1-Score will always be nearer to the smaller value of Precision or Recall.

F1-Score = (2 * Recall * Precision) / (Recall + Presision)

### Confusion Matrix (Multi-Class Classification)

![Confusion Matrix: Multi-Class Classification - TP, TN, FP, FN](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/confusion_matrix-multi-class_classification-TP_TN_FP_FN.jpg)

### True Positive (TP), False Negative (FN), True Negative (TN), False Positive (FP)

* **True Positive (TP):** Observation is positive, and is predicted to be positive.
* **False Negative (FN):** Observation is positive, but is predicted negative.
* **True Negative (TN):** Observation is negative, and is predicted to be negative.
* **False Positive (FP):** Observation is negative, but is predicted positive.

### Classification Rate / Accuracy

Accuracy = (TP + TN) / (TP + TN + FP + FN)

### Recall

Recall = TP / (TP + FN)

### Precision

Precision = TP / (TP + FP)

### F1-Score

F1-Score = (2 * Recall * Precision) / (Recall + Presision)

### Example of metrics calculation using a multi-class confusion matrix

![Confusion Matrix: Multi-Class Classification](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/confusion_matrix-multi-class_classification.png)

* True Positive (TP) of class 1: 14
* True Positive (TP) of class 2: 15
* True Positive (TP) of class 3: 6

### ACCURACY, PRECISION, RECALL, F1-SCORE FOR CLASS 1

**Accuracy (class 1)** = TP (class 1) + cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2] / sum_matrix_values

= 14 + (15 + 0 + 0 + 6) / (14 + 0 + 0 + 1 + 15 + 0 + 0 + 0 + 6) = 35/36 = 0.9722222222 (97.22 %)

**Precision (class 1)** = TP (class 1) / (cm[0][0] + cm[1][0] + cm[2][0])

= 14 / (14 + 1 + 0) = 14/15 = 0.9333333333 (93.33 %)

**Recall (class 1)** = TP (class 1) / (cm[0][0] + cm[0][1] + cm[0][2])

= 14 / (14 + 0 + 0) = 14/14 = 1.0 (100 %)

**F1-Score (class 1)** = (2 * recall_class1 * precision_class1) / (recall_class1 + precision_class1)

= (2 * 1.0 * 0.9333333333) / (1.0 + 0.9333333333) = 1.8666666666/1.9333333333 = 0.9655172414 (96.55 %)

### ACCURACY, PRECISION, RECALL, F1-SCORE FOR CLASS 2

**Accuracy (class 2)** = TP (class 2) + cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2] / sum_matrix_values: 97.22 %

**Precision (class 2)** = TP (class 2) / (cm[0][1] + cm[1][1] + cm[2][1]): 100.00 %

**Recall (class 2)** = TP (class 2) / (cm[1][0] + cm[1][1] + cm[1][2]): 93.75 %

**F1-Score (class 2)** = (2 * recall_class2 * precision_class2) / (recall_class2 + precision_class2): 96.77 %

### PRECISION, RECALL, F1-SCORE FOR CLASS 3

**Accuracy (class 3)** = TP (class 3) + cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1] / sum_matrix_values: 100.00 %

**Precision (class 3)** = TP (class 3) / (cm[0][2] + cm[1][2] + cm[2][2]): 100.00 %

**Recall (class 3)** = TP (class 3) / (cm[2][0] + cm[2][1] + cm[2][2]): 100.00 %

**F1-Score (class 3)** = (2 * recall_class3 * precision_class3) / (recall_class3 + precision_class3): 100.00 %

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