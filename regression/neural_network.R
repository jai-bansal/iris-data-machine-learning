# This script creates and evaluates a neural network in R for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering and parameter tuning.

# LOAD LIBRARIES ----------------------------------------------------------
# This section loads relevant libraries for this script.
library(readr)
library(data.table)
library(caret)

# IMPORT DATA -------------------------------------------------------------
# This section imports data.
# The data files are located in the 'R' branch file, NOT the 'classification' or 'regression' folders.
train = data.table(read_csv('train.csv'))
test = data.table(read_csv('test.csv'))

# CREATE MODEL ------------------------------------------------------------
# This section creates a neural network.

  # Set seed for reproducibility.
  set.seed(555)

  # Create neural network and include 5 fold cross validation.
  # The 'linout = T' argument is needed for regression.
  # This throws an error (not sure why), but seems to work fine in terms of prediction.
  nn = train(Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width, 
             train, 
             method = 'nnet', 
             linout = T,  
             trControl = trainControl(method = 'cv', 
                                      number = 5))
  
  # View model summary.
  nn
  
  # View a plot of hidden units, RMSE, and weight decay.
  plot(nn)
  
  # Generate training and test set predictions
  train$nn_pred = predict(nn, 
                          train)
  test$nn_pred = predict(nn, 
                         test)

# EVALUATE MODEL ----------------------------------------------------------
# Evalute 'caret' neural network by computing training, cross validation, and test set root mean square error (RMSE).
# Note that I do not include degrees of freedom in the training and test set calculations below.
# I'm not sure how 'caret' computes RMSE in the cross validation calculation.
  
  # Compute training set RMSE
  sqrt(sum((train$Sepal.Length - train$nn_pred) ^ 2) / nrow(train))
  
  # Compute cross validation set RMSE.
  mean(nn$resample$RMSE)
  
  # Compute test set RMSE.
  sqrt(sum((test$Sepal.Length - test$nn_pred) ^ 2) / nrow(test))