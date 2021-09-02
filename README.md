

<h1 align="center">Data PreProcessing</h1>

<p align="center"> EasyPreProcessing is a Python module that comprises of data pre-processing helper functions mainly for the purpose of data science and machine learning.
    <br> 
</p>

## Table of Contents

- [About](#about)
- [Installing](#install)
- [Sample Templete](#templete)

## About <a name = "about"></a>

Many of the common machine learning activities that are performed during the Feature Engineering can be performed in a single line of code using this library.

## What functionalities are currently available?
- Handling missing values
- Encoding categorical variables
- Handling DateTime features
- Handling empty/blank columns
- Display correlation metrics
- Standardize dataset
- Over sampling
- Clustering (KMeans)

## Installing <a name = "install"></a>

Just a simple

```
pip install easypreprocessing
```

## Sample Templete <a name = "templete"></a>

Below you can see a sample code of preprocessing using this library.

```
from easypreprocessing import EasyPreProcessing
prep = EasyPreProcessing('filename_here.csv')
prep.output = 'output_variable_here'

prep.remove_blank()         #Remove blank or empty columns
prep.missing_values         #Display missing values 
prep.categorical.impute()   #Fill missing values for categorical variables
prep.numerical.impute()     #Fill missing values for numerical variables
prep.categorical.encode()   #Convert categorical features to numerical
prep.standardize()          #Standardize dataset
X_train, X_test, y_train, y_test = prep.split()
```






