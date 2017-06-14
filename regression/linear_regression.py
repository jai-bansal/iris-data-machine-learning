# This script creates and evaluates a linear regression model in Python for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering and parameter tuning.

################
# IMPORT MODULES
################
# This section imports modules needed for this script.
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

#############
# IMPORT DATA
#############
# This section imports data.
# The data files are located in the 'Python' branch file, NOT the 'classification' or 'regression' folders.
# The commands below assume the working directory is set to the 'classification' folder.
os.chdir('..')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

##############
# CREATE MODEL
##############
# This section creates a linear regression model.

# Create and fit linear regression model.
linear_reg = LinearRegression().fit(train[['Sepal.Width', 'Petal.Length', 'Petal.Width']],
                                    train['Sepal.Length'])

# Conduct 5 fold cross validation.
cv = cross_val_score(linear_reg,
                     train[['Sepal.Width', 'Petal.Length', 'Petal.Width']],
                     train['Sepal.Length'],
                     cv = 5,
                     scoring = 'neg_mean_squared_error')

# View model coefficients.
linear_reg.coef_

# View intercept.
linear_reg.intercept_

# Generate training and test set predictions.
train['linear_reg_pred'] = linear_reg.predict(train[['Sepal.Width', 'Petal.Length', 'Petal.Width']])
test['linear_reg_pred'] = linear_reg.predict(test[['Sepal.Width', 'Petal.Length', 'Petal.Width']])

################
# EVALUATE MODEL
################
# This section computes the mean squared error of the model on the training, cross validation, and test sets.

# Compute mean squared error on training set.
mean_squared_error(train['Sepal.Length'],
                   train['linear_reg_pred'])

# Compute mean squared error on cross validation set.
cv.mean() * -1

# Compute mean squared error on test set.
mean_squared_error(test['Sepal.Length'],
                   test['linear_reg_pred'])
