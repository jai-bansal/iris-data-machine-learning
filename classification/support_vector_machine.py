# This script creates and evaluates a support vector machine model in Python for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering and parameter tuning.

# Import modules.
import os
import pandas as pd
from sklearn import svm, cross_validation

# Import data.
# The data files are located in the 'Python' branch file, NOT the 'classification' or 'regression' folders.
# The commands below assume the working directory is set to the 'classification' folder.
os.chdir('..')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# CREATE MODEL AND PREDICTIONS.

# Create classifier.
svm = svm.SVC(probability = False,
                         random_state = 1)

# Fit classifier on training data.
svm.fit(train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']],
       train['Species'])

# Generate predictions for training and test data.
train['pred'] = svm.predict(train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']])
test['pred'] = svm.predict(test[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']])

# CHECK TRAINING SET ACCURACY.

# Compute training set accuracy.
print('Training Set Accuracy :', sum(train['Species'] == train['pred']) / len(train['pred']))

# CROSS VALIDATION.

# Get cross validation scores.
cv_scores = cross_validation.cross_val_score(svm,
                                             train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']],
                                             train['Species'],
                                             cv = 2)

# Take the mean accuracy across all cross validation segments.
print('Cross Validation Accuracy: ', cv_scores.mean())
                                            
# CHECK TEST SET ACCURACY.

# Compute test set accuracy.
print('Test Set Accuracy :', sum(test['Species'] == test['pred']) / len(test['pred']))
