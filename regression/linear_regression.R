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

# CREATE MODEL
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
  
  
  # Generate test set predictions.
  test$linear_reg_pred = predict(linear_reg, 
                                 test)

  

# EVALUATE MODEL ----------------------------------------------------------
# This section evaluates the model by computing training, cross validation, and test set error.
  
  # Compute training set error.
  sum((train$Sepal.Length - predict(linear_reg)) ^ 2)
  
  # Compute cross validation set error.
  
  # Compute test set error.
  sum((test$Sepal.Length - test$linear_reg_pred) ^ 2)
