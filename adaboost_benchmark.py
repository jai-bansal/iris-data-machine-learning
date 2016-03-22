# This script creates and evaluates an adaboost model in Python for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering and parameter tuning.

# Import modules.
import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Import data.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# CREATE MODEL AND PREDICTIONS.

# Create classifier.
adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 5),
                              n_estimators = 100,
                              learning_rate = 1.5, 
                              random_state = 1)

# Fit classifier on training data.
adaboost.fit(train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']],
       train['Species'])

# Generate predictions for training and test data.
train['pred'] = adaboost.predict(train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']])
test['pred'] = adaboost.predict(test[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']])

# CHECK TRAINING SET ACCURACY.

# Compute training set accuracy.
print('Training Set Accuracy :', sum(train['Species'] == train['pred']) / len(train['pred']))

# CROSS VALIDATION.

# Get cross validation scores.
cv_scores = cross_validation.cross_val_score(adaboost,
                                             train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']],
                                             train['Species'],
                                             cv = 2)

# Take the mean accuracy across all cross validation segments.
print('Cross Validation Accuracy: ', cv_scores.mean())
                                            
# CHECK TEST SET ACCURACY.

# Compute test set accuracy.
print('Test Set Accuracy :', sum(test['Species'] == test['pred']) / len(test['pred']))
