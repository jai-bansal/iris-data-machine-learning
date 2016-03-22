# This script creates and evaluates a K Nearest Neighbor model in Python for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering and parameter tuning.

# Import modules.
import pandas as pd
from sklearn import preprocessing, cross_validation
from sklearn.neighbors import KNeighborsClassifier

# Import data.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# SCALE DATA.

# Create scaler for each feature.
scaler = preprocessing.StandardScaler().fit(train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']])

# CREATE MODEL AND PREDICTIONS.

# Create classifier.
knn = KNeighborsClassifier(n_neighbors = 5)

# Fit classifier on scaled training data.
knn.fit(scaler.transform(train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]),
       train['Species'])

# Generate predictions for (scaled) training and test data.
train['pred'] = knn.predict(scaler.transform(train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]))
test['pred'] = knn.predict(scaler.transform(test[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]))

# CHECK TRAINING SET ACCURACY.

# Compute training set accuracy.
print('Training Set Accuracy :', sum(train['Species'] == train['pred']) / len(train['pred']))

# CROSS VALIDATION.

# Get cross validation scores.
cv_scores = cross_validation.cross_val_score(knn,
                                             scaler.transform(train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]),
                                             train['Species'],
                                             cv = 2)

# Take the mean accuracy across all cross validation segments.
print('Cross Validation Accuracy: ', cv_scores.mean())
                                            
# CHECK TEST SET ACCURACY.

# Compute test set accuracy.
print('Test Set Accuracy :', sum(test['Species'] == test['pred']) / len(test['pred']))
