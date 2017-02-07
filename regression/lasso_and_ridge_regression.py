# This script creates and evaluates LASSO and ridge regression models in Python for the 'iris' data.
# This iteration is only to get the models up and running, so there is no/minimal feature engineering or parameter tuning.

################
# IMPORT MODULES
################
# This section imports modules needed for this script.
import os
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso, RidgeCV, Ridge
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

###############################
# CREATE LASSO REGRESSION MODEL
###############################
# This section creates a LASSO regression model.

# Create and fit LASSO regression model with 3 fold cross validation to determine alpha.
lasso_cv = LassoCV(cv = 3,
                   random_state = 555).fit(train[['Sepal.Width', 'Petal.Length', 'Petal.Width']],
                                           train['Sepal.Length'])

# View the amount of penalization chosen by cross validation.
lasso_cv.alpha_

# View coefficients.
lasso_cv.coef_

# View intercept.
lasso_cv.intercept_

# Conduct 3 fold cross validation with 'lasso_cv.alpha_' to get cross validation error below.
best_lasso_cv = cross_val_score(Lasso(alpha = lasso_cv.alpha_,
                                      random_state = 555),
                                train[['Sepal.Width', 'Petal.Length', 'Petal.Width']],
                                train['Sepal.Length'],
                                cv = 3,
                                scoring = 'neg_mean_squared_error')

# Generate training and test set predictions.
train['lasso_pred'] = lasso_cv.predict(train[['Sepal.Width', 'Petal.Length', 'Petal.Width']])
test['lasso_pred'] = lasso_cv.predict(test[['Sepal.Width', 'Petal.Length', 'Petal.Width']])

###############################
# CREATE RIDGE REGRESSION MODEL
###############################
# This section creates a ridge regression model.

# Create and fit ridge regression model with 3 fold cross validation to determine alpha.
ridge_cv = RidgeCV(cv = 3).fit(train[['Sepal.Width', 'Petal.Length', 'Petal.Width']],
                               train['Sepal.Length'])

# View coefficients.
ridge_cv.coef_

# View intercept.
ridge_cv.intercept_

# View estimated regularization parameter.
ridge_cv.alpha_

# Conduct 3 fold cross validation with 'lasso_cv.alpha_' to get cross validation error below.
best_ridge_cv = cross_val_score(Ridge(alpha = ridge_cv.alpha_),
                                train[['Sepal.Width', 'Petal.Length', 'Petal.Width']],
                                train['Sepal.Length'],
                                cv = 3,
                                scoring = 'neg_mean_squared_error')

# Generate training and test set predictions.
train['ridge_pred'] = ridge_cv.predict(train[['Sepal.Width', 'Petal.Length', 'Petal.Width']])
test['ridge_pred'] = ridge_cv.predict(test[['Sepal.Width', 'Petal.Length', 'Petal.Width']])

#################################
# EVALUATE LASSO REGRESSION MODEL
#################################
# This section computes the mean squared error of the LASSO regression model on the training, cross validation, and test sets.

# Compute mean squared error on training set.
mean_squared_error(train['Sepal.Length'],
                   train['lasso_pred'])

# Compute mean squared error on cross validation set.
best_lasso_cv.mean() * -1

# Compute mean squared error on test set.
mean_squared_error(test['Sepal.Length'],
                   test['lasso_pred'])

#################################
# EVALUATE RIDGE REGRESSION MODEL
#################################
# This section computes the mean squared error of the ridge regression model on the training, cross validation, and test sets.

# Compute mean squared error on training set.
mean_squared_error(train['Sepal.Length'],
                   train['ridge_pred'])

# Compute mean squared error on cross validation set.
best_ridge_cv.mean() * -1

# Compute mean squared error on test set.
mean_squared_error(test['Sepal.Length'],
                   test['ridge_pred'])
