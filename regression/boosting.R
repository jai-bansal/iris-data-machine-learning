# This script creates and evaluates a boosting regression model in R for the 'iris' data set.
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

# CREATE MODEL ------------------------------------------------------------
# This section creates a boosting regression model.

  # Set seed for reproducibility.
  set.seed(555)
  
  # Create boosting regression model and include 5 fold cross validation.
  boosting_reg = train(Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width, 
                       train, 
                       method = 'gbm', 
                       trControl = trainControl(method = 'cv', 
                                                number = 5))

  # View model summary.
  boosting_reg
  
  # View variable importance plot.
  plot(varImp(boosting_reg))
  
  # Generate training and test set predictions
  train$boost_pred = predict(boosting_reg, 
                             train)
  test$boost_pred = predict(boosting_reg, 
                            test)

# EVALUATE MODEL ----------------------------------------------------------
# This section evaluates the model by computing training, cross validation, and test set root mean square error (RMSE).
# Note that I do not include degrees of freedom in the training and test set calculations below.
# I'm not sure how 'caret' computes RMSE in the cross validation calculation.

  # Compute training set RMSE
  sqrt(sum((train$Sepal.Length - train$boost_pred) ^ 2) / nrow(train))
  
  # Compute cross validation set RMSE.
  mean(boosting_reg$resample$RMSE)
  
  # Compute test set RMSE.
  sqrt(sum((test$Sepal.Length - test$boost_pred) ^ 2) / nrow(test))