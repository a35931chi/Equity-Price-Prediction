## Introduction

This is an attempt to try to predict stock prices in the future. I want to explore and compare the results between LSTM and other modeling algorithms. I also want to explore the possibilities of optimizing LSTMs. Analysis done in python IDLE 3.5.3. 
Please refer to the [blank](blank) and [blank](blank) for the complete project and report details.


## Please refer to:
* [Project Proposal](https://github.com/a35931chi/MLND-Capstone/blob/master/capstone_proposal.pdf): Detailing project motivation
* [Project Report](https://github.com/a35931chi/MLND-Capstone/blob/master/capstone_report.pdf): Detailing feature engineering and modeling excercise
* [Feature Engineering](https://github.com/a35931chi/MLND-Capstone/blob/master/Code/Feature_Engineering.py): Code for feature engineering
* [Main Model Optimization & Selection](https://github.com/a35931chi/MLND-Capstone/blob/master/Code//Model_Optimization_GridSearchCV.py): Code for model optimization using GridSearchCV
* [Visualizations](https://github.com/a35931chi/MLND-Capstone/tree/master/Visualization): Folder contains all visualizations for the project
* [Data Dictionary](https://github.com/a35931chi/MLND-Capstone/blob/master/Feature_Engineering.xlsx): Data dictionary
* [Model Summary](https://github.com/a35931chi/MLND-Capstone/blob/master/Model_Selection.xlsx): Selecting optimized modeling algorithm

## References:
### Utility Libraries:
```
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import visuals as vs #from one of the MLND project. Was useful in visualizing PCA
from time import gmtime, strftime, time
import warnings
```

### Statistical Libraries:
```
from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import norm, skew 
```
### Modeling Libraries:
```
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split, ShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
```
### Data Gathering Library:
```
from Feature_Engineering import data_wrangle
```
