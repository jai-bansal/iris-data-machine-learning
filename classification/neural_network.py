# This script creates and evaluates a neural network classification model in Python for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering or parameter tuning.

################
# IMPORT MODULES
################
# This section imports modules needed for this script.
import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

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
# This section creates a neural network classification model.

# Create and fit neural network regression model.
nn_class = MLPClassifier(max_iter = 700,
                         random_state = 555).fit(train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']],
                                                 train['Species'])

# View current loss.
nn_class.loss_

# View number of iterations.
nn_class.n_iter_

# View number of layers.
nn_class.n_layers_

# View name of output activation function.
nn_class.out_activation_

# Conduct 3 fold cross validation.
cv = cross_val_score(nn_class,
                     train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']],
                     train['Species'],
                     cv = 3,
                     scoring = 'accuracy')

# Generate training and test set predictions.
train['nn_class_pred'] = nn_class.predict(train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']])
test['nn_class_pred'] = nn_class.predict(test[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']])

################
# EVALUATE MODEL
################
# This section computes the mean squared error of the model on the training, cross validation, and test sets.

# Compute mean squared error on training set.
sum(train['Species'] == train['nn_class_pred']) / len(train['nn_class_pred'])

# Compute mean squared error on cross validation set.
cv.mean()

# Compute mean squared error on test set.
sum(test['Species'] == test['nn_class_pred']) / len(test['nn_class_pred'])
