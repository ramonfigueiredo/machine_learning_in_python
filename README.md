Machine Learning in Python
===========================

## Data Preprocessing
1. [Data Preprocessing](#data-preprocessing)
2. [How to the python program](#how-to-the-python-program)

## Data Preprocessing

1. Data Preprocessing
	* Importing the libraries
	* Taking care of missing data
	* Encoding categorical data
	* Splitting the dataset into the Training set and Test set
	* Feature Scaling

	a.  [data_preprocessing.py](https://github.com/ramonfigueiredopessoa/machine_learning_in_python/blob/master/src/1_data_preprocessing/data_preprocessing.py)

## How to run the python program?

1. Install [virtualenv](https://virtualenv.pypa.io/en/latest/) and [Flask](https://palletsprojects.com/p/flask/)
	* To activate the virtualenv on Linux or MacOS: ```source venv/bin/activate```
	* To activate the virtualenv on Windows: ```\venv\Script\activate.bat```

## How to the python program

Step-by-step:

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