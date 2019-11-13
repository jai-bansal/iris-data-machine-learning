# This script demos LIME for a "scikit-learn" classification model.

################
# IMPORT MODULES
################
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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
rf = RandomForestClassifier(n_estimators = 101,
                               oob_score = True,
                               random_state = 555).fit(train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']],
                                                       train['Species'])

######
# LIME
######
# LIME only seems to work in Jupyter notebooks!

# Regression
explainer = lime.lime_tabular.LimeTabularExplainer(training_data = np.array(train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]), 
                                                   mode = "classification",
                                                   feature_names = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'], 
                                                   class_names = ["Species"], 
                                                   verbose = True)

exp = explainer.explain_instance(test[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']].iloc[0], 
                                 rf.predict_proba, num_features = 3)

# The plot isn't perfect, but can work with some more interpretation
exp.show_in_notebook(show_table=True, show_all=False)

explainer.explain_instance()

