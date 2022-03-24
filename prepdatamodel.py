import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

#If we want to add some new atributes, else just comment it
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6 #positions of atributes, which are used to create new ones
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):#creating a class to use it like sklearn classes
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        # Here we write our new atributes calculations
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
#two tipes of num pipelines with standart diviation and MinMax normalization
num_pipeline_std = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

num_pipeline_mmx = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('mmx_scaler', MinMaxScaler())
])

def pipeline(dataset, num_attributes, cat_attributes, std = True):
    if std:
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline_std, num_attributes),
            ("cat", OneHotEncoder(), cat_attributes),
            ])
    else:
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline_mmx, num_attributes),
            ("cat", OneHotEncoder(), cat_attributes),
            ])
    return full_pipeline.fit_transform(dataset)