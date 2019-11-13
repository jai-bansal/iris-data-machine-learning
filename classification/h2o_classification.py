# This script shows some capabilities of the H2O platform.
# I train a few classification models.
# I also include grid search and auto-ML.

# LIME is a little tricky to get working for H2O models and so is not included.

################
# IMPORT MODULES
################
import pandas as pd

import h2o
from h2o.estimators import H2ORandomForestEstimator, H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.automl import H2OAutoML

import lime.lime_tabular
import numpy as np

##############
# IMPORT DATA
#############
# The data files are located in the 'Python' branch file, NOT the 'classification' or 'regression' folders.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#####
# H2O
#####
# This section shows H2O capabilities.

h2o.init()                             # Initialize cluster

# Create H2O data frames
h2o_train = h2o.H2OFrame(train)
h2o_test = h2o.H2OFrame(test)

# Set independent and dependent variables
y = "Species"
x = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]

# Models

# GLM
glm_classifier = H2OGeneralizedLinearEstimator(family = "multinomial")
glm_classifier.train(y = y, x = x, training_frame = h2o_train)

# Random forest
rf = H2ORandomForestEstimator(ntrees = 50, max_depth = 10)
rf.train(y = y, x = x, training_frame = h2o_train)

# XGBoost
gbm = H2OGradientBoostingEstimator(ntrees = 50, max_depth = 10)
gbm.train(y = y, x = x, training_frame = h2o_train)

# Performance
glm_classifier.model_performance(test_data = h2o_test)
rf.model_performance(test_data = h2o_test)
gbm.model_performance(test_data = h2o_test)

# Predictions
h2o_preds = rf.predict(h2o_test)
h2o_preds = h2o.as_list(h2o_preds)

# Grid search
params = {'ntrees': [50, 100], 'max_depth': [5, 10]}                  # Define search space

grid_search = H2OGridSearch(model = H2ORandomForestEstimator, 
                            grid_id = "grid1", 
                            hyper_params = params)
grid_search.train(y = y, x = x, training_frame = h2o_train)
r = grid_search.get_grid(sort_by = 'rmse', decreasing = True)       # Results

# AutoML (seems like AutoML can't be run right after another AutoML)
auto = H2OAutoML(max_runtime_secs_per_model = 30)
auto.train(y = y, x = x, training_frame = h2o_train)
h2o.as_list(auto.leaderboard)   