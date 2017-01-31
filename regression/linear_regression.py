# This script creates and evaluates a linear regression model in Python for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering and parameter tuning.

################
# IMPORT MODULES
################
# This section imports modules needed for this script.
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

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
# This section creates a linear regression model.

# Create and fit linear regression model.
linear_reg = LinearRegression().fit(train[['Sepal.Width', 'Petal.Length', 'Petal.Width']],
                                    train['Sepal.Length'])

# View model coefficients.
linear_reg.coef_

# View intercept.
linear_reg.intercept_

# Generate training and test set predictions.
train['linear_reg_pred'] = linear_reg.predict(train[['Sepal.Width', 'Petal.Length', 'Petal.Width']])
test['linear_reg_pred'] = linear_reg.predict(test[['Sepal.Width', 'Petal.Length', 'Petal.Width']])
