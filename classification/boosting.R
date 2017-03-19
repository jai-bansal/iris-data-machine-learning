# This script creates and evaluates a few boosting models in R for the 'iris' data set.
# This iteration is only to get the models up and running, so there is no feature engineering or parameter tuning.

# LOAD LIBRARIES ----------------------------------------------------------
# This section loads relevant libraries for this script.
library(readr)
library(data.table)
library(adabag)
library(caret)
library(xgboost)

# IMPORT DATA -------------------------------------------------------------
# This section imports data.
# The data files are located in the 'R' branch file, NOT the 'classification' or 'regression' folders.
train = data.table(read_csv('train.csv'))
test = data.table(read_csv('test.csv'))

# CLEAN AND PROCESS DATA --------------------------------------------------
# This section prepares the data for modeling.

  # Change 'Species' to 'factor' for 'train' and 'test'.
  train$Species = as.factor(train$Species)
  test$Species = as.factor(test$Species)
  
  # 'xgboost' needs the label to range from 0 to the # of classes minus 1.
  # Transform data to this format.
  
    # Transform 'train$Species' to match this format.
    train[, 
          species_transform := .GRP, 
          by = 'Species']
    train$species_transform = (train$species_transform) - 1
    
    # Make sure 'test$Species' is in the exact same format.
    test = merge(test, 
                 unique(train[, .(Species, species_transform)]), 
                 by = 'Species')

# 'adabag' PACKAGE MODEL --------------------------------------------------
# This section implements a boosting model using the 'adabag' package.

  # Set seed to ensure reproducibility.
  set.seed(1)
  
  # Define model.
  model = Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
  
  # Create boosting model.
  adabag_boost = boosting(model, 
                      data = train, 
                      boos = F,
                      mfinal = 10,
                      control = rpart.control(cp = -1))
  
  # Generate predictions for training and test data.
  # Training set predictions were computed when model was trained.
  train$adabag_pred = adabag_boost$class
  test$adabag_pred = data.table(predict(adabag_boost, 
                                        newdata = test)$class)
  
  # View training set accuracy.
  print('Training Accuracy:')
  table(train$Species == train$adabag_pred)
  prop.table(table(train$Species == train$adabag_pred))
  
  # Cross validation.

    # Conduct cross validation.
    adabag_cv = boosting.cv(model, 
                            data = train, 
                            v = 3, 
                            boos = F, 
                            mfinal = 10, 
                            control = rpart.control(cp = -1))
    
    # View estimated test set error computed by cross validation.
    print('Cross Validation Accuracy: ')
    print(1 - adabag_cv$error)

  # View test set accuracy.
  print('Test Set Accuracy:')
  table(test$Species == test$adabag_pred)
  prop.table(table(test$Species == test$adabag_pred))
  
# 'caret' PACKAGE MODEL ---------------------------------------------------
# This section implements a boosting model using the 'caret' package.
  
  # Set seed to ensure reproducibility.
  set.seed(1)
  
  # Train boosting model using 'caret' package.
  caret_boost = train(model, 
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
  
  # View training set accuracy.
  print('Training Accuracy:')
  table(train$Species == train$caret_pred)
  prop.table(table(train$Species == train$caret_pred))
  
  # View cross validation set accuracy.
  print('Cross Validation Accuracy')
  mean(caret_boost$resample$Accuracy)
  
  # View test set accuracy.
  print('Test Set Accuracy:')
  table(test$Species == test$caret_pred)
  prop.table(table(test$Species == test$caret_pred))

# 'xgboost' PACKAGE MODEL -------------------------------------------------
# This section builds a boosting model using the 'xgboost' package.
  
  # Set seed to ensure reproducibility.
  set.seed(1)

  # Train model using 'xgboost' package.
  xg_boost = xgboost(data = as.matrix(train[, .(Sepal.Length, Sepal.Width, Petal.Length, Petal.Width)]), 
                     label = train$species_transform, 
                     params = list(num_class = 3, 
                                   objective = 'multi:softmax'), 
                     nrounds = 10)
  
  # View evaluation log of model.
  xg_boost$evaluation_log
  
  # View model parameters.
  xg_boost$params
  
  # Conduct cross validation.
  xg_boost_cv = xgb.cv(data = as.matrix(train[, .(Sepal.Length, Sepal.Width, Petal.Length, Petal.Width)]), 
                       label = train$species_transform, 
                       params = list(num_class = 3, 
                                     objective = 'multi:softmax'), 
                       nrounds = 10, 
                       nfold = 3)
  
  # Generate predictions for training and test data.
  train$xgboost_pred = predict(xg_boost, 
                               newdata = as.matrix(train[, .(Sepal.Length, Sepal.Width, Petal.Length, Petal.Width)]))
  test$xgboost_pred = predict(xg_boost, 
                              newdata = as.matrix(test[, .(Sepal.Length, Sepal.Width, Petal.Length, Petal.Width)]))
  
  # View training set accuracy.
  print('Training Accuracy:')
  table(train$species_transform == train$xgboost_pred)
  prop.table(table(train$species_transform == train$xgboost_pred))
  
  # View mean cross validation error.
  print('Cross Validation Accuracy')
  1 - mean(xg_boost_cv$evaluation_log$test_merror_mean)
  
  # View test set accuracy.
  print('Test Set Accuracy:')
  table(test$species_transform == test$xgboost_pred)
  prop.table(table(test$species_transform == test$xgboost_pred))