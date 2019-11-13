# This script demos LIME for a "scikit-learn" regression model.

################
# IMPORT MODULES
################
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import lime.lime_tabular
import numpy as np

##############
# IMPORT DATA
#############
# The data files are located in the 'Python' branch file, NOT the 'classification' or 'regression' folders.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

##############
# CREATE MODEL
##############
# This section creates a random forest regression model.

# Create and fit random forest regression model.
rf_reg = RandomForestRegressor(n_estimators = 101,
                               oob_score = True,
                               random_state = 555).fit(train[['Sepal.Width', 'Petal.Length', 'Petal.Width']],
                                                       train['Sepal.Length'])

######
# LIME
######
# LIME only seems to work in Jupyter notebooks!

# Regression
explainer = lime.lime_tabular.LimeTabularExplainer(training_data = np.array(train[['Sepal.Width', 'Petal.Length', 'Petal.Width']]), 
                                                   mode = "regression",
                                                   feature_names = ['Sepal.Width', 'Petal.Length', 'Petal.Width'], 
                                                   class_names = ["r_cont"], 
                                                   verbose = True)

exp = explainer.explain_instance(test[['Sepal.Width', 'Petal.Length', 'Petal.Width']].iloc[0], 
                                 rf_reg.predict, num_features = 3)

exp.show_in_notebook(show_table=True, show_all=False)

