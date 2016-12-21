# This script creates and evaluates a random forest model in R for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering or parameter tuning.
# I include implementations in the 'randomForest', 'rrf' (regularized random forest), and 'caret' packages.

# LOAD LIBRARIES ----------------------------------------------------------
# This section loads relevant libraries.
library(data.table)
library(randomForest)
library(rrf)
library(caret)

# IMPORT DATA -------------------------------------------------------------
# This section imports data.
train = data.table(read.csv('train.csv',
                            header = T))
test = data.table(read.csv('test.csv', 
                           header = T))

# 'randomForest' PACKAGE MODEL --------------------------------------------
# This section implements a random forest model using the 'randomForest' package.

  # Define model.
  model = Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
  
  # Set seed for reproducibility.
  set.seed(1)
  
  # Create random forest.
  rf = randomForest(model, 
                    data = train, 
                    ntree = 10, 
                    importance = T)
  
  # View feature importance.
  varImpPlot(rf)
  
  # Generate predictions for training and test data and put into new data tables.
  train_pred = data.table(rf_pred = predict(rf, 
                                            newdata = train, 
                                            type = 'response'))
  test_pred = data.table(rf_pred = predict(rf, newdata = test, 
                                           type = 'response'))
 
  # Check training set accuracy.
  table(train$Species == train_pred$rf_pred)
  prop.table(table(train$Species == train_pred$rf_pred))
  
  # Check OOB score (used for cross validation in random forests).
  table(train$Species == rf$predicted)
  prop.table(table(train$Species == rf$predicted))
  
  # Check test set accuracy.
  table(test$Species == test_pred$rf_pred)
  prop.table(table(test$Species == test_pred$rf_pred))
  

# 'rrf' PACKAGE MODEL -----------------------------------------------------
# This section implements a random forest model using the 'rrf' package (regularized random forest).
# There may not be much difference between 'rrf' and 'randomForest' results in this data set because the
# feature space is small (6 features).
  
  # Set seed for reproducibility.
  set.seed(1)
  
  # Create regularized random forest.
  rrf = randomForest(model, 
                     data = train, 
                     ntree = 10, 
                     importance = T)
  
  # View feature importance.
  varImpPlot(rrf)
  
  # Generate predictions for training and test data.
  train_pred$rrf_pred = data.table(predict(rrf, 
                                           newdata = train, 
                                           type = 'response'))
  test_pred$rrf_pred = data.table(predict(rrf, 
                                          newdata = test, 
                                          type = 'response'))
  
  # Check training set accuracy.
  table(train$Species == train_pred$rrf_pred)
  prop.table(table(train$Species == train_pred$rrf_pred))
  
  # Check OOB score (used for cross validation in random forests).
  table(train$Species == rrf$predicted)
  prop.table(table(train$Species == rrf$predicted))
  
  # Check test set accuracy.
  table(test$Species == test_pred$rrf_pred)
  prop.table(table(test$Species == test_pred$rrf_pred))

  
# 'caret' PACKAGE MODEL ---------------------------------------------------
# This section implements a random forest model using the 'caret' package.

  