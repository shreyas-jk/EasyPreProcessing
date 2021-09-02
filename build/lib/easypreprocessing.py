import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from kneed import KneeLocator
from imblearn.over_sampling import SMOTE, RandomOverSampler
from IPython.display import Markdown, display

class EasyPreProcessing:
    """
    Create an object of type EasyPreProcessing
      
    Attributes:
        path (str):
        dataset (DataFrame): 
        output (str): 
        categorical (CategoryFeatures): 
        numerical (NumericFeatures):
        clustering (Clustering): 
        dates (DateTimeFeatures): 
    """
    def __init__(self, path):
        self.path = path
        self.dataset = self.load_dataframe(path)
        self.output = None
        self.categorical = CategoryFeatures(self)
        self.numerical = NumericFeatures(self)
        self.clustering = Clustering(self)
        self.dates = DateTimeFeatures(self)
        printmd("""
**Initialization Parameters**

1.  output            - Set output variable/dependent variable
2.  dates.features    - Set datetime field names (optional)

For example:
1.  output = 'column_name'
2.  dates.features = ['date_field_1','date_field_2']
        """)
    
    @property
    def data(self):
        return self.dataset
    
    @property
    def df(self):
        return self.dataset
    
    @property
    def missing_values(self):
        return self.dataset.isnull().sum()
    
    @property
    def columns(self):
        return self.dataset.columns

    @property
    def info(self):
        printmd("""

**General Template**
<code>
from easypreprocessing import EasyPreProcessing
prep = EasyPreProcessing('filename_here.csv')
prep.df
prep.output = 'output_variable_here'
prep.remove_blank()
prep.missing_values
prep.categorical.impute()
prep.numerical.impute()
prep.categorical.encode()
prep.correction()
prep.standardize()
X_train, X_test, y_train, y_test = prep.split()
</code>


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

**Numerical Preprocessing**
<table>
<tr>
<td>numerical.fields</td><td>Display all numerical field names</td>
</tr>
<tr>
<td>numerical.impute()</td><td>Handle numerical missing values. Parameters {'mean', 'medium', 'mode'}</td>
</tr>
</table>

**Date Preprocessing**
<table>
<tr>
<td>dates.features</td><td>Define list of all datetime feature names</td>
</tr>
<tr>
<td>dates.split_datetime()</td><td>Split all datetime features into discrete fields (Year, Month, Day, Hour, Minute)</td>
</tr>
</table>

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
        """)

    def load_dataframe(self, path):
        return pd.read_csv(path)
    
    @property
    def fields(self):
        return self.dataset.columns

    def reinit(self):
        self.dataset = self.load_dataframe(self.path)
    
    def get_X(self):
        return self.dataset[self.dataset.columns.difference([self.output])]

    def get_Y(self):
        return self.dataset[self.output]

    def split(self, test_size = 0.2, random_state = 0):
        self.X = self.get_X().values
        self.y = self.get_Y().values
        return train_test_split(self.X, self.y, test_size = test_size, random_state = random_state)

    def standardize(self):
        self.dataset[self.__get_unique_numerical()] = pd.DataFrame(StandardScaler().fit_transform(self.dataset[self.__get_unique_numerical()]))
        print('Dataset has been standardized.')
    
    def correction(self):
        sns.heatmap(self.dataset.corr())
    
    def remove_blank(self):
        emply_columns = [col for col in self.dataset.columns if self.dataset[col].isnull().all()]
        self.dataset = self.dataset.drop(emply_columns, axis=1)

    def __get_unique_numerical(self):
        return [col for col in list(self.dataset.columns) if (self.dataset[col].dtype in ['float64', 'int64', 'int32']) and self.output != col]
    
    def encode_output(self):
        self.dataset[self.output] = pd.DataFrame(preprocessing.LabelEncoder().fit_transform(self.dataset[self.output]))
        print('Label encoded the output variable.')

    def over_sample(self, strategy='random'):
        """
        This method will over sample your dataset to balance your output variable/dependent feature.
  
        Parameters: 
            strategy (str):  {'random': RandomOverSampling, 'smote':SMOTE }. Default random
          
        Returns: 
            None
        """
        if strategy == 'smote':
            print('Sampling method selected: SMOTE')
            smt = SMOTE()
            X_train, y_train = smt.fit_resample(self.get_X(), self.get_Y())
            self.dataset = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
        if strategy == 'random':
            print('Sampling method selected: RandomOverSampling')
            ros = RandomOverSampler()
            X_train, y_train = ros.fit_resample(self.get_X(), self.get_Y())
            self.dataset = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
        print('Oversampling completing.')

class DateTimeFeatures:
    def __init__(self, parent):
        self.parent = parent
        self.features = []
    
    def split_datetime(self, drop=True):
        """
        This method will split all datetime features into discrete fields (Year, Month, Day, Hour, Minute).
  
        Parameters: 
            drop (bool):  Drop original datetime field. Default True
          
        Returns: 
            None
        """
        for col in self.features:
            print('Splitting ', col, '...')
            self.parent.dataset[col + '_Year'] = pd.to_datetime(self.parent.dataset[col]).dt.year
            self.parent.dataset[col + '_Month'] = pd.to_datetime(self.parent.dataset[col]).dt.month
            self.parent.dataset[col + '_Day'] = pd.to_datetime(self.parent.dataset[col]).dt.day
            self.parent.dataset[col + '_Hour'] = pd.to_datetime(self.parent.dataset[col]).dt.hour
            self.parent.dataset[col + '_Minute'] = pd.to_datetime(self.parent.dataset[col]).dt.minute
            if drop == True:
                self.parent.dataset = self.parent.dataset.drop([col], axis = 1)
                print('Dropped column: ', col)
        print('Newly added columns: Year, Month, Day, Hour, Minute')

class Clustering():
    def __init__(self, parent):
        self.parent = parent

    def elbow_plot(self, elbow_limit, init, random_state):
        wcss=[]
        for i in range (1, elbow_limit):
            kmeans=KMeans(n_clusters=i, init=init, random_state=random_state)
            kmeans.fit(self.parent.dataset)
            wcss.append(kmeans.inertia_)
        return KneeLocator(range(1, elbow_limit), wcss, curve='convex', direction='decreasing')
    
    def apply(self, elbow_limit=11, init='k-means++', random_state=42):
        """
        Apply KMeans clustering to entire dataset. "Cluster" column will be appended to the existing dataset that will mark which cluster does the record belong.
  
        Parameters: 
            elbow_limit (int):  Elbow limit is required for finding the best K value for clustering. Default 11 or greater.
            init (int):         Default 'k-means++'
            random_state (int): Default 42
          
        Returns: 
            None
        """
        print('Calculate best Knee value for clustering...')
        self.no_clusters = self.elbow_plot(elbow_limit, init, random_state)
        print('Started clustering process...')
        self.kmeans = KMeans(n_clusters=self.no_clusters.knee, init=init, random_state=random_state)
        self.kmeans_clusters_list = self.kmeans.fit_predict(self.parent.dataset)
        self.parent.dataset['Cluster'] = self.kmeans_clusters_list
        print('Number of clusters created:', len(np.unique(self.kmeans_clusters_list)))

class NumericFeatures():
    def __init__(self, parent):
        self.parent = parent

    @property
    def fields(self):
        """
        Get list of all names of numerical columns/features.
  
        Parameters: 
            None
          
        Returns: 
            list
        """
        return self.__get_unique_numerical()
    
    def __get_unique_numerical(self):
        return [col for col in list(self.parent.dataset.columns) if self.parent.dataset[col].dtype == 'float64' or self.parent.dataset[col].dtype == 'int64']
       
    def impute(self, strategy='mean'):
        """
        Method to impute numerical features.
  
        Parameters:
            strategy (str): {'mean', 'median', 'mode'}. Default 'mean'
          
        Returns:
            None
        """
        imputer = SimpleImputer(missing_values = np.nan, strategy = strategy)
        self.parent.dataset[self.__get_unique_numerical()] = imputer.fit_transform(self.parent.dataset[self.__get_unique_numerical()])
        print('Numerical features imputated successfully.')

class CategoryFeatures():
    def __init__(self, parent):
        self.parent = parent
    
    @property
    def fields(self):
        """
        Get list of all names of categorical columns/features.
  
        Parameters: 
            None
          
        Returns: 
            list
        """
        return self.__get_unique_categorical()

    @property
    def unique(self):
        """
        Print unique or distinct values of all categorical features.
  
        Parameters: 
            None
          
        Returns: 
            None
        """
        for category in self.__get_unique_categorical():
            unique_values = self.parent.dataset[category].unique()
            print(str(category) + ' ' + str(unique_values))
        
    def __get_unique_categorical(self):
        return [col for col in list(self.parent.dataset.columns) if self.parent.dataset[col].dtype == 'object' and self.parent.output != col]
    
    def get_feature_mode(self, column):
        return self.parent.dataset[column].mode().values[0]

    def impute(self):
        """
        Method to impute numerical features. Strategy 'mode' by default.
          
        Returns: 
            None
        """
        for col in list(self.__get_unique_categorical()):
            mode = self.get_feature_mode(col)
            self.parent.dataset[col].fillna(mode, inplace=True)
        print('Categorical features imputated successfully.')
    
    def label_encoder(self):
        self.parent.dataset[self.__get_unique_categorical()] = self.parent.dataset[self.__get_unique_categorical()].apply(preprocessing.LabelEncoder().fit_transform)

    def onehotencoder(self):
        encoded_data = pd.get_dummies( self.parent.dataset.drop([self.parent.output], axis=1) )
        self.parent.dataset = pd.concat( [encoded_data , self.parent.dataset[self.parent.output] ] , axis=1)

    def encode(self, strategy='ohe'):
        """
        Encoding categorical features using LabelEncoding or OneHotEncoding.
  
        Parameters:
            strategy (str): {'ohe','le'}. Default 'ohe'.
                            'ohe': OneHotENcoding
                            'le': LabelEnoding
          
        Returns:
            None
        """
        if strategy == 'ohe':
            self.onehotencoder()
            print('Dataset has been succesfully encoded using OneHotEncoder')
        if strategy == 'le':
            self.label_encoder()
            print('Dataset has been succesfully encoded using LabelEncoder')

def printmd(string):
    display(Markdown(string))

