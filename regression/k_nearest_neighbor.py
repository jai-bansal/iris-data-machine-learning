# This script creates and evaluates a k nearest neighbor (KNN) regression model in Python for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering and parameter tuning.

################
# IMPORT MODULES
################
# This section imports modules needed for this script.
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
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

####################
# CREATE DATA SCALER
####################
# This section creates a scaler for the data in preparation for the KNN regression model.
scaler = StandardScaler().fit(train[['Sepal.Width', 'Petal.Length', 'Petal.Width']])

##############
# CREATE MODEL
##############
# This section creates a KNN regression model.

# Create KNN regression model using 5 nearest neighbors.
# In an actual modeling application, I would optimize the number of nearest neighbors.
# But in this script, I just want to get the model up and running.
knn_reg = KNeighborsRegressor(n_neighbors = 5)

# Fit KNN regression model.
knn_reg.fit(scaler.transform(train[['Sepal.Width', 'Petal.Length', 'Petal.Width']]),
            train['Sepal.Length'])

# Conduct 3 fold cross validation.
cv = cross_val_score(knn_reg,
                     scaler.transform(train[['Sepal.Width', 'Petal.Length', 'Petal.Width']]),
                     train['Sepal.Length'],
                     cv = 3,
                     scoring = 'neg_mean_squared_error')

# Generate training and test set predictions.
train['knn_reg_pred'] = knn_reg.predict(scaler.transform(train[['Sepal.Width', 'Petal.Length', 'Petal.Width']]))
test['knn_reg_pred'] = knn_reg.predict(scaler.transform(test[['Sepal.Width', 'Petal.Length', 'Petal.Width']]))

################
# EVALUATE MODEL
################
# This section computes the mean squared error of the model on the training, cross validation, and test sets.

# Compute mean squared error on training set.
mean_squared_error(train['Sepal.Length'],
                   train['knn_reg_pred'])

# Compute mean squared error on cross validation set.
cv.mean() * -1

# Compute mean squared error on test set.
mean_squared_error(test['Sepal.Length'],
                   test['knn_reg_pred'])
