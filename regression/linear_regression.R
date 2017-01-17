# This script creates and evaluates a linear regression model in R for the 'iris' data.
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
# This section creates a linear regression model.

  # Set seed for reproducibility.
  set.seed(555)

  # Create linear regression model and include 5 fold cross validation.
  linear_reg = train(Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width, 
                     train, 
                     method = 'lm', 
                     trControl = trainControl(method = 'cv', 
                                              number = 5))
  
  # View model summary.
  summary(linear_reg)
  
  # View variable importance plot.
  plot(varImp(linear_reg))
  
  # Generate training and test set predictions
  train$linear_reg_pred = predict(linear_reg, train)
  test$linear_reg_pred = predict(linear_reg, 
                                 test)

# EVALUATE MODEL ----------------------------------------------------------
# This section evaluates the model by computing training, cross validation, and test set root mean square error (RMSE).
# Note that I do not include degrees of freedom in the training and test set calculations below.
# I'm not sure how 'caret' computes RMSE in the cross validation calculation.
  
  # Compute training set RMSE
  sqrt(sum((train$Sepal.Length - predict(linear_reg)) ^ 2) / nrow(train))
  
  # Compute cross validation set RMSE.
  mean(linear_reg$resample$RMSE)
  
  # Compute test set RMSE.
  sqrt(sum((test$Sepal.Length - test$linear_reg_pred) ^ 2) / nrow(test))