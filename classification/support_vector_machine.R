# This script creates and evaluates a support vector machine model in R for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering and parameter tuning.

# Load packages.
library(readr)
library(data.table)
library(e1071)

# IMPORT DATA --------------------------------------------------------------
# This section imports data.
# The data files are located in the 'R' branch file, NOT the 'classification' or 'regression' folders.
# The commands below assume the working directory is set to the 'classification' folder.
train = data.table(read_csv('../train.csv'))
test = data.table(read_csv('../test.csv'))

#####
# CREATE MODEL AND PREDICTIONS.

  # Set seed to ensure reproducibility.
  set.seed(1)

  # Define model.
  model = Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
  
  # Create SVM.
  svm = svm(model, 
            data = train, 
            kernel = 'linear', 
            probability = T,
            cost = 0.01, 
            scale = T)

  # View indices of observations that are support vectors.
  svm$index
  
  # View length of support vector index vector.
  print(length(svm$index))
  
  # Generate predictions for training and test data.
  train$pred = data.table(predict(svm,
                               newdata = train,
                               decision.values = F))
  test$pred = predict(svm, 
                      newdata = test, 
                      decision.values = T)

#####
# CHECK TRAINING SET ACCURACY.

  # View training accuracy.
  print('Training Accuracy:')
  table(train$Species == train$pred)
  prop.table(table(train$Species == train$pred))
    
#####
# CROSS VALIDATION.
# Estimate test set error using cross validation.
    
    # Create 'tune.control' object.
    tune_control = tune.control(cross = 2)
    
    # Execute cross validation.
    t = tune('svm', 
             model, 
             data = train, 
             tunecontrol = tune_control)
    
    # View estimated test set error computed by cross validation.
    print(t)

#####
# TEST SET ACCURACY.

  # View test set accuracy.
  print('Test Set Accuracy:')
  table(test$Species == test$pred)
  prop.table(table(test$Species == test$pred))
