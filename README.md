

<h1 align="center">Data PreProcessing</h1>

<p align="center"> EasyPreProcessing is a Python module that comprises of data pre-processing helper functions mainly for the purpose of data science and machine learning.
    <br> 
</p>

## Table of Contents

- [About](#about)
- [Installing](#install)
- [Sample Templete](#templete)

# About <a name = "about"></a>

Many of the common machine learning activities that are performed during the Feature Engineering can be performed in a single line of code using this library.

# What functionalities are currently available?
- Handling missing values
- Encoding categorical variables
- Handling DateTime features
- Handling empty/blank columns
- Display correlation metrics
- Standardize dataset
- Over sampling
- Clustering (KMeans)
<br> 
<br> 

# Installing <a name = "install"></a>

Just a simple

```
pip install easypreprocessing
```

For details regarding all the functionality available:

```
from easypreprocessing import EasyPreProcessing
prep = EasyPreProcessing('filename.csv')
prep.info()
```

**Categorical Preprocessing**
<table>
<tr>
<td>categorical.fields</td><td>Display all categorical field names</td>
</tr>
<tr>
<td>categorical.unique</td><td>Display unique/distinct categorical values</td>
</tr>
<tr>
<td>categorical.impute()</td><td>Handle categorical missing values. Parameters {'mean', 'medium', 'mode'}</td>
</tr>
<tr>
<td>categorical.encode()</td><td>Encode categorical features. Parameters {'le': LabelEncoding, 'ohe': OneHotEncoding}</td>
</tr>
</table>
<br> 

**Numerical Preprocessing**
<table>
<tr>
<td>numerical.fields</td><td>Display all numerical field names</td>
</tr>
<tr>
<td>numerical.impute()</td><td>Handle numerical missing values. Parameters {'mean', 'medium', 'mode'}</td>
</tr>
</table>
<br> 

**Date Preprocessing**
<table>
<tr>
<td>dates.features</td><td>Define list of all datetime feature names</td>
</tr>
<tr>
<td>dates.split_datetime()</td><td>Split all datetime features into discrete fields (Year, Month, Day, Hour, Minute)</td>
</tr>
</table>
<br> 

**General Preprocessing**
<table>
<tr>
<td>missing_values</td><td>Display missing value report</td>
</tr>
<tr>
<td>remove_blank()</td><td>Remove empty/blank columns</td>
</tr>
<tr>
<td>correction()</td><td>Display correction heatmap</td>
</tr>
<tr>
<td>standardize()</td><td>Standardize entire dataset except dependent variable</td>
</tr>
<tr>
<td>encode_output()</td><td>Encode dependent feature/output variable</td>
</tr>
<tr>
<td>over_sample()</td><td>Oversample dataset. Parameters {'smote': SMOTE, 'ros': RandomOverSample}</td>
</tr>
<tr>
<td>clustering.apply()</td><td>Cluster dataset using elbow plot</td>
</tr>
</table>

<br> 
<br> 

# Sample Templete <a name = "templete"></a>

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






