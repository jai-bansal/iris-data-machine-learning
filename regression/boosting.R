# This script creates and evaluates boosting regression models in R for the 'iris' data set.
# This iteration is only to get the models up and running, so there is no feature engineering or parameter tuning.

# LOAD LIBRARIES ----------------------------------------------------------
# This section loads relevant libraries for this script.
library(readr)
library(data.table)
library(caret)
library(xgboost)

# IMPORT DATA -------------------------------------------------------------
# This section imports data.
# The data files are located in the 'R' branch file, NOT the 'classification' or 'regression' folders.
train = data.table(read_csv('train.csv'))
test = data.table(read_csv('test.csv'))

# 'caret' PACKAGE MODEL ---------------------------------------------------
# This section implements a boosting model using the 'caret' package.
  
  # Set seed to ensure reproducibility.
  set.seed(1)
  
  # Train boosting model using 'caret' package.
  caret_boost = train(Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width, 
                      train, 
                      method = 'gbm', 
                      trControl = trainControl(method = 'cv', 
                                               number = 3))

  # View model summary.
  caret_boost
  
  # View variable importance plot.
  plot(varImp(caret_boost))
  
  # Generate training and test set predictions
  train$caret_pred = predict(caret_boost, 
                             train)
  test$caret_pred = predict(caret_boost, 
                            test)
  
  # View training set RMSE.
  print('Training RMSE:')
  sqrt(sum((train$Sepal.Length - train$caret_pred) ^ 2) / nrow(train))
  
  # View cross validation set RMSE.
  print('Cross Validation RMSE')
  mean(caret_boost$resample$RMSE)
  
  # View test set RMSE.
  print('Test Set RMSE:')
  sqrt(sum((test$Sepal.Length - test$caret_pred) ^ 2) / nrow(test))
  
# 'xgboost' PACKAGE MODEL -------------------------------------------------
# This section builds a boosting model using the 'xgboost' package.
  
  # Set seed to ensure reproducibility.
  set.seed(1)

  # Train model using 'xgboost' package.
  xg_boost = xgboost(data = as.matrix(train[, .(Sepal.Width, Petal.Length, Petal.Width)]), 
                     label = train$Sepal.Length, 
                     params = list(objective = 'reg:linear'), 
                     nrounds = 10)
  
  # View evaluation log of model.
  xg_boost$evaluation_log
  
  # View model parameters.
  xg_boost$params
  
  # Conduct cross validation.
  xg_boost_cv = xgb.cv(data = as.matrix(train[, .(Sepal.Width, Petal.Length, Petal.Width)]), 
                     label = train$Sepal.Length, 
                     params = list(objective = 'reg:linear'), 
                     nrounds = 10, 
                     nfold = 3)
  
  # Generate predictions for training and test data.
  train$xgboost_pred = predict(xg_boost, 
                               newdata = as.matrix(train[, .(Sepal.Width, Petal.Length, Petal.Width)]))
  test$xgboost_pred = predict(xg_boost, 
                              newdata = as.matrix(test[, .(Sepal.Width, Petal.Length, Petal.Width)]))
  
  # Compute training set RMSE
  sqrt(sum((train$Sepal.Length - train$xgboost_pred) ^ 2) / nrow(train))
  
  # View mean cross validation RMSE.
  # This seems wrong...magnitude is much different than training or test RMSE.
  mean(xg_boost_cv$evaluation_log$test_rmse_mean)
  
  # Compute test set RMSE.
  sqrt(sum((test$Sepal.Length - test$xgboost_pred) ^ 2) / nrow(test))
