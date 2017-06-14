# This script creates and evaluates a neural network regression in Python for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering or parameter tuning.

################
# IMPORT MODULES
################
# This section imports modules needed for this script.
import os
import pandas as pd
from sklearn.neural_network import MLPRegressor
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
# This section creates a neural network regression model.

# Create and fit neural network regression model.
nn_reg = MLPRegressor(max_iter = 500,
                      random_state = 555).fit(train[['Sepal.Width', 'Petal.Length', 'Petal.Width']],
                                              train['Sepal.Length'])

# View current loss.
nn_reg.loss_

# View the number of iterations carried out.
nn_reg.n_iter_

# View the number of layers.
nn_reg.n_layers_

# View name of output activation function (check documentation).
nn_reg.out_activation_

# Conduct 3 fold cross validation.
cv = cross_val_score(nn_reg,
                     train[['Sepal.Width', 'Petal.Length', 'Petal.Width']],
                     train['Sepal.Length'],
                     cv = 3,
                     scoring = 'neg_mean_squared_error')

# Generate training and test set predictions.
train['nn_reg_pred'] = nn_reg.predict(train[['Sepal.Width', 'Petal.Length', 'Petal.Width']])
test['nn_reg_pred'] = nn_reg.predict(test[['Sepal.Width', 'Petal.Length', 'Petal.Width']])

################
# EVALUATE MODEL
################
# This section computes the mean squared error of the model on the training, cross validation, and test sets.

# Compute mean squared error on training set.
mean_squared_error(train['Sepal.Length'],
                   train['nn_reg_pred'])

# Compute mean squared error on cross validation set.
cv.mean() * -1

# Compute mean squared error on test set.
mean_squared_error(test['Sepal.Length'],
                   test['nn_reg_pred'])
