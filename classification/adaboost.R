# This script creates and evaluates an Adaboost model in R for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering and parameter tuning.

# Load packages.
library(data.table)
library(adabag)

# Import data.
train = data.table(read.csv('train.csv',
                            header = T))
test = data.table(read.csv('test.csv', 
                           header = T))

#####
# CREATE MODEL AND PREDICTIONS.
  
  # Set seed to ensure reproducibility.
  set.seed(1)
  
  # Define model.
  model = Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
  
  # Create model (using 'subset').
  adaboost = boosting(model, 
                      data = train, 
                      boos = F,
                      mfinal = 10,
                      control = rpart.control(cp = -1))
  
  # Generate predictions for training and test data.
  # Training set predictions were computed when model was trained.
  train$pred = adaboost$class
  test_pred = predict(adaboost, 
                      newdata = test)
  test$pred = data.table(predict(adaboost, 
                                 newdata = test)$class)

#####
# TRAINING SET ACCURACY.
  
  # View training accuracy.
  print('Training Accuracy:')
  table(train$Species == train$pred)
  prop.table(table(train$Species == train$pred))

#####
# CROSS VALIDATION.
    
  # Conduct cross validation.
  cv = boosting.cv(model, 
                   data = train, 
                   v = 2, 
                   boos = F, 
                   mfinal = 10, 
                   control = rpart.control(cp = -1))
  
  # View estimated test set error computed by cross validation.
  print('Cross Validation Error: ')
  print(cv$error)

#####
# TEST SET ACCURACY.

  # View test set accuracy.
  print('Test Set Accuracy:')
  table(test$Species == test$pred)
  prop.table(table(test$Species == test$pred))