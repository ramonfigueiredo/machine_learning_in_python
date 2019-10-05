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
8. [How to run the Python program](#how-to-run-the-python-program)

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

### Metrics: 

* **True Positive (TP):** Observation is positive, and is predicted to be positive.
* **False Negative (FN):** Observation is positive, but is predicted negative.
* **True Negative (TN):** Observation is negative, and is predicted to be negative.
* **False Positive (FP):** Observation is negative, but is predicted positive.

#### Classification Rate / Accuracy

Classification Rate or Accuracy is given by the relation:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

However, there are problems with accuracy.  It assumes equal costs for both kinds of errors. A 99% accuracy can be excellent, good, mediocre, poor or terrible depending upon the problem.

#### Recall

Recall can be defined as the ratio of the total number of correctly classified positive examples divide to the total number of positive examples. High Recall indicates the class is correctly recognized (small number of FN).

Recall is given by the relation:

Recall = TP / (TP + FN)

#### Precision

To get the value of precision we divide the total number of correctly classified positive examples by the total number of predicted positive examples. High Precision indicates an example labeled as positive is indeed positive (small number of FP).

Precision is given by the relation:

Precision = TP / (TP + FP)

High recall, low precision: 
This means that most of the positive examples are correctly recognized (low FN) but there are a lot of false positives.

Low recall, high precision: 
This shows that we miss a lot of positive examples (high FN) but those we predict as positive are indeed positive (low FP)

#### F-measure

Since we have two measures (Precision and Recall) it helps to have a measurement that represents both of them. We calculate an F-measure which uses Harmonic Mean in place of Arithmetic Mean as it punishes the extreme values more.

The F-Measure will always be nearer to the smaller value of Precision or Recall.

Fmeasure = (2 * Recall * Precision) / (Recall + Presision)

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