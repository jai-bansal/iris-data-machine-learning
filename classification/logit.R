# This script creates and evaluates a multinomial logistic regression model in R for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering and parameter tuning.

# Load packages.
library(data.table)
library(nnet)

# Import data.
# The data files are located in the 'R' branch file, NOT the 'classification' or 'regression' folders.
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
  
  # Create model and generate predictions for training set.
  multi_logit = multinom(model, 
              data = train)
  
  # Generate predictions for training and test data.
  train$pred = data.table(predict(multi_logit, 
                                  newdata = train, 
                                  type = 'class'))
  test$pred = data.table(predict(multi_logit, 
                                 newdata = test, 
                                 type = 'class'))

#####
# CHECK TRAINING SET ACCURACY.

  # View training accuracy.
  print('Training Accuracy:')
  table(train$Species == train$pred)
  prop.table(table(train$Species == train$pred))

#####
# CROSS VALIDATION.
# I couldn't find a cross validation function for this technique so I write my own.
  
  # Specify number of folds.
  folds = 2
  
  # Create random row assignment vector.
  train$assignment = sample(1:folds, 
                            replace = T, 
                            nrow(train))
  
  # Create vector to hold cross validation results.
  cv_results = rep(NA, folds)
  
  # Conduct cross validation.
  for (i in 1:folds)
    
    {
    
      # Set training set as all observations with 'assignment == i'.
      train_cv = subset(train, 
                        train$assignment == i)

      # Set test set as all observations with 'assignment != i'.
      test_cv = subset(train, 
                    train$assignment != i)

      # Train model on training data.
      ml = multinom(model, 
                    data = train_cv)
      
      # Generate predictions for test set and append to 'test'.
      test_cv$pred = data.table(predict(ml, 
                                     newdata = test_cv, 
                                     type = 'class'))

      # Add test set error to 'cv_results'.
      cv_results[i] = nrow(subset(test_cv, 
                           test_cv$Species == test_cv$pred)) / nrow(test_cv)
    
    }
  
  # Print cross validation error.
  print('Cross Validation Accuracy:')
  mean(cv_results)
 
#####
# EXPORT TEST SET PREDICTIONS.

  # View test set accuracy.
  print('Test Set Accuracy:')
  table(test$Species == test$pred)
  prop.table(table(test$Species == test$pred))
