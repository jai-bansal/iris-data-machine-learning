# This script creates and evaluates a multinomial logistic regression model in Python for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering and parameter tuning.

# Import modules.
import pandas as pd
from sklearn import linear_model, cross_validation

# Import data.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# CREATE MODEL AND PREDICTIONS.

# Create classifier.
multinom_logit = linear_model.LogisticRegression(random_state = 1)

# Fit classifier on training data.
multinom_logit.fit(train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']],
       train['Species'])

# Generate predictions for training and test data.
train['pred'] = multinom_logit.predict(train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']])
test['pred'] = multinom_logit.predict(test[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']])

# CHECK TRAINING SET ACCURACY.

# Compute training set accuracy.
print('Training Set Accuracy :', sum(train['Species'] == train['pred']) / len(train['pred']))

# CROSS VALIDATION.

# Get cross validation scores.
cv_scores = cross_validation.cross_val_score(multinom_logit,
                                             train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']],
                                             train['Species'],
                                             cv = 2)

# Take the mean accuracy across all cross validation segments.
print('Cross Validation Accuracy: ', cv_scores.mean())
                                            
# CHECK TEST SET ACCURACY.

# Compute test set accuracy.
print('Test Set Accuracy :', sum(test['Species'] == test['pred']) / len(test['pred']))
