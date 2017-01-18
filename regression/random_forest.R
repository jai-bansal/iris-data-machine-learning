# This script creates and evaluates random forest regression models in R for the 'iris' data set.
# This iteration is only to get models up and running, so there is no feature engineering and parameter tuning.
# I include implementations in the 'randomForest', 'rrf' (regularized random forest), and 'caret' packages.

# LOAD LIBRARIES ----------------------------------------------------------
# This section loads relevant libraries for this script.
library(readr)
library(data.table)
library(randomForest)
library(RRF)
library(caret)

# IMPORT DATA -------------------------------------------------------------
# This section imports data.
# The data files are located in the 'R' branch file, NOT the 'classification' or 'regression' folders.
train = data.table(read_csv('train.csv'))
test = data.table(read_csv('test.csv'))

# 'randomForest' PACKAGE MODEL --------------------------------------------
# This section implements a random forest regression model using the 'randomForest' package.

  # Specify model.
  model = Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width

  # Set seed for reproducibility.
  set.seed(555)
  
  # Create random forest model using 'randomForest' package.
  rf_randomForest = randomForest(model, 
                                 data = train, 
                                 ntree = 101, 
                                 importance = T)
  
  # View model summary.
  rf_randomForest
  
  # View feature importance.
  varImp(rf_randomForest)
  
  # Generate predictions for training and test data.
  train$randomForest_pred = predict(rf_randomForest, 
                                    newdata = train)
  test$randomForest_pred = predict(rf_randomForest, 
                                   newdata = test)
  
  # Evaluate 'randomForest' random forest model by computing training, cross validation, and test set root mean square error (RMSE).
  # Note that I do not include degrees of freedom in the calculations below.
  
    # Compute training set RMSE
    sqrt(sum((train$Sepal.Length - train$randomForest_pred) ^ 2) / nrow(train))
    
    # Compute cross validation set RMSE.
    # I use out-of-bag (OOB) samples.
    sqrt(sum((train$Sepal.Length - rf_randomForest$predicted) ^ 2) / nrow(train))
    
    # Compute test set RMSE.
    sqrt(sum((test$Sepal.Length - test$randomForest_pred) ^ 2) / nrow(test))

# 'RRF' PACKAGE MODEL -----------------------------------------------------
# This section implements a random forest regression model using the 'RRF' package.
# There may not be much difference between 'RRF' and 'randomForest' results in this data set because the
# feature space is small.
  
  # Set seed for reproducibility.
  set.seed(555)
  
  # Create random forest model using 'RRF' package.
  rf_RRF = RRF(model, 
               data = train, 
               ntree = 101, 
               importance = T)
  
  # View model summary.
  rf_RRF
  
  # View feature importance.
  varImp(rf_RRF)
  
  # Generate predictions for training and test data.
  train$RRF_pred = predict(rf_RRF, 
                           newdata = train)
  test$RRF_pred = predict(rf_RRF, 
                          newdata = test)
  
  # Evaluate 'RRF' random forest model by computing training, cross validation, and test set root mean square error (RMSE).
  # Note that I do not include degrees of freedom in the calculations below.
  
    # Compute training set RMSE
    sqrt(sum((train$Sepal.Length - train$RRF_pred) ^ 2) / nrow(train))
    
    # Compute cross validation set RMSE.
    # I use out-of-bag (OOB) samples.
    sqrt(sum((train$Sepal.Length - rf_RRF$predicted) ^ 2) / nrow(train))
    
    # Compute test set RMSE.
    sqrt(sum((test$Sepal.Length - test$RRF_pred) ^ 2) / nrow(test))
  

# 'caret' PACKAGE MODEL ------------------------------------------------------------
# This section implements a random forest regression model using the 'caret' package.
  
  # Set seed for reproducibility.
  set.seed(555)

  # Create random forest regression model and include 5 fold cross validation.
  rf_caret = train(model, 
                   train, 
                   method = 'rf',
                   trControl = trainControl(method = 'cv', 
                                            number = 5))
  
  # View model summary.
  rf_caret
  
  # Generate training and test set predictions
  train$caret_pred = predict(rf_caret, 
                             train)
  test$caret_pred = predict(rf_caret, 
                            test)
  
  # Evalute 'caret' random forest model by computing training, cross validation, and test set root mean square error (RMSE).
  # Note that I do not include degrees of freedom in the training and test set calculations below.
  # I'm not sure how 'caret' computes RMSE in the cross validation calculation.
  
  # Compute training set RMSE
  sqrt(sum((train$Sepal.Length - train$caret_pred) ^ 2) / nrow(train))
  
  # Compute cross validation set RMSE.
  mean(rf_caret$resample$RMSE)
  
  # Compute test set RMSE.
  sqrt(sum((test$Sepal.Length - test$caret_pred) ^ 2) / nrow(test))