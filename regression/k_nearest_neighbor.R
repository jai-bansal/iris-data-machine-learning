# This script creates and evaluates a k-nearest-neighbor (KNN) regression model in R for the 'iris' data set.
# This iteration is only to get the model up and running, so there is no feature engineering or parameter tuning.

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

# SCALE DATA --------------------------------------------------------------
# This section scales the data for the KNN model below.

  # Scale relevant columns of 'train'.
  train_scale = scale(train[, .(Sepal.Width, Petal.Length, Petal.Width)])

  # Save means and standard deviations of 'train'.
  train_means = attr(train_scale, 
                     'scaled:center')
  train_sd = attr(train_scale, 
                  'scaled:scale')
  
  # Turn 'train_scale' to 'data.table' and add 'train$Sepal.Length'.
  train_scale = data.table(train_scale)
  train_scale$Sepal.Length = train$Sepal.Length

  # Scale relevant columns of 'test' according to the scaling done for 'train'.
  test_scale = data.table(Sepal.Width = (test$Sepal.Width - train_means[1]) / train_sd[1], 
                          Petal.Length = (test$Petal.Length - train_means[2]) / train_sd[2], 
                          Petal.Width = (test$Petal.Width - train_means[3]) / train_sd[3])
  
  # Add 'test$Sepal.Length' to 'test_scale'.
  test_scale$Sepal.Length = test$Sepal.Length

# CREATE MODEL ------------------------------------------------------------
# This section creates a KNN regression model.

  # Set seed for reproducibility.
  set.seed(555)

  # Create KNN regression model with 5 clusters and include 5 fold cross validation.
  knn_reg = train(Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width, 
                  train_scale, 
                  method = 'knn', 
                  tuneGrid = expand.grid(k = 5),
                  trControl = trainControl(method = 'cv', 
                                           number = 5))
  
  # View model summary.
  knn_reg
  
  # View variable importance plot.
  plot(varImp(knn_reg))
  
  # Generate training and test set predictions
  train_scale$knn_pred = predict(knn_reg, 
                                 train_scale)
  test_scale$knn_pred = predict(knn_reg, 
                                test_scale)

# EVALUATE MODEL ----------------------------------------------------------
# This section evaluates the model by computing training, cross validation, and test set root mean square error (RMSE).
# Note that I do not include degrees of freedom in the training and test set calculations below.
# I'm not sure how 'caret' computes RMSE in the cross validation calculation.
# I'm also not sure cross validation is a great idea for KNN due to sample size concerns.
  
  # Compute training set RMSE
  sqrt(sum((train_scale$Sepal.Length - train_scale$knn_pred) ^ 2) / nrow(train_scale))
  
  # Compute cross validation set RMSE.
  mean(knn_reg$resample$RMSE)
  
  # Compute test set RMSE.
  sqrt(sum((test_scale$Sepal.Length - test_scale$knn_pred) ^ 2) / nrow(test_scale))