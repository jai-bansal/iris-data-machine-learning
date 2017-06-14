# This script creates and evaluates a random forest model in Python for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering and parameter tuning.

# Import modules.
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Import data.
# The data files are located in the 'Python' branch file, NOT the 'classification' or 'regression' folders.
# The commands below assume the working directory is set to the 'classification' folder.
os.chdir('..')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# CREATE MODEL AND PREDICTIONS.

# Create classifier.
rf = RandomForestClassifier(n_estimators = 25,
                            max_depth = 15,
                            oob_score = True,
                            random_state = 1)

# Fit classifier on training data.
rf.fit(train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']],
       train['Species'])

# View feature importances.
print('Feature Importances: ', rf.feature_importances_)

# Generate predictions for training and test data.
train['pred'] = rf.predict(train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']])
test['pred'] = rf.predict(test[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']])

# CHECK TRAINING SET ACCURACY.
# I'll check OOB (out of bag) error estimate and model performance on the training data set.

# View OOB score.
# For random forests, OOB score makes cross validation unnecessary.
print('OOB Accuracy: ', rf.oob_score_)

# Compute training set accuracy.
print('Training Set Accuracy :', sum(train['Species'] == train['pred']) / len(train['pred']))

# CHECK TEST SET ACCURACY.

# Compute test set accuracy.
print('Test Set Accuracy :', sum(test['Species'] == test['pred']) / len(test['pred']))
