Machine Learning in Python
===========================

## Data Preprocessing
1. [Data Preprocessing](#data-preprocessing)
2. [Regression](#regression)
	1. [Simple Linear Regression](#simple-linear-regression)
3. [How to run the Python program](#how-to-run-the-python-program)

## Data Preprocessing

### Data Preprocessing
	* Taking care of missing data
	* Encoding categorical data
	* Splitting the dataset into the Training set and Test set
	* Feature Scaling

	a.  [data_preprocessing.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/1_data_preprocessing/data_preprocessing.py)

## Regression

### Simple Linear Regression
	* Importing the dataset (Salary_Data.csv)
	* Splitting the dataset into the Training set and Test set
	* Fitting Simple Linear Regression to the Training set
	* Predicting the Test set results
	* Visualising the Training set results (matplotlib: scatter plot)
	![Visualising the Training set resultsr](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/1_simple_linear_regression/Visualising-the-Test-set-results.png)
	* Visualising the Test set results (matplotlib: scatter plot)
	![Visualising the Training set resultsr](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/1_simple_linear_regression/Visualising-the-Test-set-results.png)

	a.  [simple_linear_regression.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/2_regression/1_simple_linear_regression/simple_linear_regression.py)


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