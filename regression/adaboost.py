# This script creates and evaluates an adaboost regression model in Python for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering and parameter tuning.

################
# IMPORT MODULES
################
# This section imports modules needed for this script.
import os
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

#############
# IMPORT DATA
#############
# This section imports data.
# The data files are located in the 'Python' branch file, NOT the 'classification' or 'regression' folders.

# Set working directory (this obviously only works on my personal machine).
os.chdir('D:\\Users\JBansal\Documents\GitHub\iris-data-machine-learning')

# Import data.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

##############
# CREATE MODEL
##############
# This section creates an adaboost regression model.

# Create and fit adaboost regression model.
adaboost_reg = AdaBoostRegressor(n_estimators = 101,
                                 learning_rate = 1,
                                 random_state = 555).fit(train[['Sepal.Width', 'Petal.Length', 'Petal.Width']],
                                                         train['Sepal.Length'])

# View feature importances.
adaboost_reg.feature_importances_

# Conduct 3 fold cross validation.
cv = cross_val_score(adaboost_reg,
                     train[['Sepal.Width', 'Petal.Length', 'Petal.Width']],
                     train['Sepal.Length'],
                     cv = 3,
                     scoring = 'neg_mean_squared_error')

# Generate training and test set predictions.
train['adaboost_reg_pred'] = adaboost_reg.predict(train[['Sepal.Width', 'Petal.Length', 'Petal.Width']])
test['adaboost_reg_pred'] = adaboost_reg.predict(test[['Sepal.Width', 'Petal.Length', 'Petal.Width']])

################
# EVALUATE MODEL
################
# This section computes the mean squared error of the model on the training, cross validation, and test sets.

# Compute mean squared error on training set.
mean_squared_error(train['Sepal.Length'],
                   train['adaboost_reg_pred'])

# Compute mean squared error on cross validation set.
cv.mean() * -1

# Compute mean squared error on test set.
mean_squared_error(test['Sepal.Length'],
                   test['adaboost_reg_pred'])
