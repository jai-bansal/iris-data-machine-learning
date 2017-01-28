# This script creates and evaluates a boosting regression model in R for the 'iris' data set.
# This iteration is only to get the model up and running, so there is no feature engineering or parameter tuning.

# LOAD LIBRARIES ----------------------------------------------------------
# This section loads relevant libraries for this script.
library(readr)
library(data.table)
library(gbm)

# IMPORT DATA -------------------------------------------------------------
# This section imports data.
# The data files are located in the 'R' branch file, NOT the 'classification' or 'regression' folders.
train = data.table(read_csv('train.csv'))
test = data.table(read_csv('test.csv'))

# CREATE MODEL ------------------------------------------------------------
# This section creates a boosting regression model.

  # Set seed for reproducibility.
  set.seed(555)

  # Create KNN regression model with 5 clusters and include 5 fold cross validation.
  boosting_reg = gbm(formula = Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width, 
                     
                     data = train)
  
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