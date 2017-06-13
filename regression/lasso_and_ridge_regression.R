# This script creates and evaluates LASSO and ridge regression models in R for the 'iris' data.
# This iteration is only to get the models up and running, so there is no feature engineering or parameter tuning.

# LOAD LIBRARIES ----------------------------------------------------------
# This section loads relevant libraries for this script.
library(readr)
library(data.table)
library(caret)
library(glmnet)

# IMPORT DATA -------------------------------------------------------------
# This section imports data.
# The data files are located in the 'R' branch file, NOT the 'classification' or 'regression' folders.
train = data.table(read_csv('train.csv'))
test = data.table(read_csv('test.csv'))

# CREATE LASSO REGRESSION MODEL ------------------------------------------------------------
# This section creates a LASSO regression model.

  # Conduct 5 fold cross validation for LASSO regression to find the best penalty coefficient.
  lasso_cv = cv.glmnet(x = as.matrix(train[, .(Sepal.Width, Petal.Length, Petal.Width)]), 
                       y = train$Sepal.Length, 
                       family = 'gaussian', 
                       alpha = 1, 
                       nfolds = 5)

  # View plot of penalty coefficients, mean-squared error, and number of non-zero coefficients.
  plot(lasso_cv)

  # View optimal penalty coefficient.
  lasso_cv$lambda.min

  # Generate training and test set predictions
  train$lasso_pred = predict(lasso_cv, 
                             newx = as.matrix(train[, .(Sepal.Width, Petal.Length, Petal.Width)]))
  test$lasso_pred = predict(lasso_cv, 
                            newx = as.matrix(test[, .(Sepal.Width, Petal.Length, Petal.Width)]))


# CREATE RIDGE REGRESSION MODEL -------------------------------------------
# This section creates a ridge regression model.
  
  # Conduct 5 fold cross validation for ridge regression to find the best penalty coefficient.
  ridge_cv = cv.glmnet(x = as.matrix(train[, .(Sepal.Width, Petal.Length, Petal.Width)]), 
                       y = train$Sepal.Length, 
                       family = 'gaussian', 
                       alpha = 0, 
                       nfolds = 5)

  # View plot of penalty coefficients, mean-squared error, and number of non-zero coefficients.
  plot(ridge_cv)

  # View optimal penalty coefficient.
  ridge_cv$lambda.min

  # Generate training and test set predictions
  train$ridge_pred = predict(ridge_cv, 
                             newx = as.matrix(train[, .(Sepal.Width, Petal.Length, Petal.Width)]))
  test$ridge_pred = predict(ridge_cv, 
                            newx = as.matrix(test[, .(Sepal.Width, Petal.Length, Petal.Width)]))


# EVALUATE LASSO MODEL ----------------------------------------------------------
# This section evaluates the LASSO regression model above by computing training, cross validation, and test set root mean square error (RMSE).
# Note that I do not include degrees of freedom in the training and test set calculations below.
# For cross validation error, I report 'mean cross-validated error' (from 'glmnet' documentation), but I don't know exactly how this is computed.

  # Compute training set RMSE
  sqrt(sum((train$Sepal.Length - train$lasso_pred) ^ 2) / nrow(train))
  
  # Compute cross validation set RMSE.
  lasso_cv$cvm[match(lasso_cv$lambda.min, lasso_cv$lambda)]
  
  # Compute test set RMSE.
  sqrt(sum((test$Sepal.Length - test$lasso_pred) ^ 2) / nrow(test))

# EVALUATE RIDGE REGRESSION MODEL -----------------------------------------
# This section evaluates the LASSO regression model above by computing training, cross validation, and test set root mean square error (RMSE).
# Note that I do not include degrees of freedom in the training and test set calculations below.
# For cross validation error, I report 'mean cross-validated error' (from 'glmnet' documentation), but I don't know exactly how this is computed.
  
  # Compute training set RMSE
  sqrt(sum((train$Sepal.Length - train$ridge_pred) ^ 2) / nrow(train))
  
  # Compute cross validation set RMSE.
  ridge_cv$cvm[match(ridge_cv$lambda.min, ridge_cv$lambda)]
  
  # Compute test set RMSE.
  sqrt(sum((test$Sepal.Length - test$ridge_pred) ^ 2) / nrow(test))