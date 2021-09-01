import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from kneed import KneeLocator

class EasyPreProcessing:
    def __init__(self, path):
        self.path = path
        self.dataset = self.load_dataframe(path)
        self.output = None
        self.categorical = CategoryFeatures(self)
        self.numerical = NumericFeatures(self)
    
    def load_dataframe(self, path):
        return pd.read_csv(path)
    
    def reinit(self):
        self.dataset = self.load_dataframe(self.path)
    
    def get_data(self):
        return self.dataset
    
    def split(self, test_size = 0.2, random_state = 0):
        self.X = self.dataset[self.dataset.columns.difference([self.output])].values
        self.y = self.dataset[self.output].values
        return train_test_split(self.X, self.y, test_size = test_size, random_state = random_state)
    
    def show_missing(self):
        return self.dataset.isnull().sum()
    
    def standardize(self):
        self.dataset[self.__get_unique_numerical()] = pd.DataFrame(StandardScaler().fit_transform(self.dataset[self.__get_unique_numerical()]))
        
    def show_correction(self):
        sns.heatmap(self.dataset[self.__get_unique_numerical()].corr())
    
    def __get_unique_numerical(self):
        return [col for col in list(self.dataset.columns) if self.dataset[col].dtype == 'float64' or self.dataset[col].dtype == 'int64']


class Clustering():
    def __init__(self, parent):
        self.parent = parent

    def elbow_plot(self, elbow_limit, init, random_state):
        wcss=[]
        for i in range (1, elbow_limit):
            kmeans=KMeans(n_clusters=i, init=init, random_state=random_state)
            kmeans.fit(self.parent.dataset)
            wcss.append(kmeans.inertia_)
        self.no_clusters = KneeLocator(range(1, elbow_limit), wcss, curve='convex', direction='decreasing')
    
    def apply(self, elbow_limit=10, init='k-means++', random_state=42):
        self.elbow_plot(elbow_limit, init, random_state)
        self.kmeans = KMeans(n_clusters=self.no_clusters, init=init, random_state=random_state)
        self.y_kmeans = self.kmeans.fit_predict(self.parent.dataset)
        self.parent.dataset['Cluster'] = self.y_kmeans
        print('Number of clusters created:', np.unique(self.y_kmeans))

class NumericFeatures():
    def __init__(self, parent):
        self.parent = parent

    def names(self):
        return self.__get_unique_numerical()
    
    def __get_unique_numerical(self):
        return [col for col in list(self.parent.dataset.columns) if self.parent.dataset[col].dtype == 'float64' or self.parent.dataset[col].dtype == 'int64']
       
    def impute(self, strategy='mean'):
        imputer = SimpleImputer(missing_values = np.nan, strategy = strategy)
        self.parent.dataset[self.__get_unique_numerical()] = imputer.fit_transform(self.parent.dataset[self.__get_unique_numerical()])


class CategoryFeatures():
    def __init__(self, parent):
        self.parent = parent
        
    def __get_unique_categorical(self):
        return [col for col in list(self.parent.dataset.columns) if self.parent.dataset[col].dtype == 'object']
    
    def names(self):
        return self.__get_unique_categorical()
    
    def get_feature_mode(self, column):
        return self.parent.dataset[column].mode().values[0]

    def unique(self):
        for category in self.__get_unique_categorical():
            unique_values = self.parent.dataset[category].unique()
            print(str(category) + ' ' + str(unique_values))

    def impute(self):
        for col in list(self.__get_unique_categorical()):
            mode = self.get_feature_mode(col)
            self.parent.dataset[col].fillna(mode, inplace=True)

    def label_encoder(self):
        self.parent.dataset[self.__get_unique_categorical()] = self.parent.dataset[self.__get_unique_categorical()].apply(preprocessing.LabelEncoder().fit_transform)

    def onehotencoder(self):
        self.parent.dataset = pd.get_dummies(self.parent.dataset)

    def encode(self, type='ohe'):
        if type == 'ohe':
            self.onehotencoder()
        if type == 'le':
            self.label_encoder()



