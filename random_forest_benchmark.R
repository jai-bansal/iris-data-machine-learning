# This script creates and evaluates a random forest model in R for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering and parameter tuning.

# Load packages.
library(data.table)
library(randomForest)

# Import data.
train = data.table(read.csv('train.csv',
                            header = T))
test = data.table(read.csv('test.csv', 
                           header = T))

#####
# CREATE MODEL AND PREDICTIONS.

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
  
  # Generate predictions for training and test data.
  train$pred = data.table(predict(rf, 
                                  newdata = train, 
                                  type = 'response'))
  test$pred = data.table(predict(rf, 
                                 newdata = test, 
                                 type = 'response'))
  
#####
# CHECK TRAINING SET ACCURACY.
# I'll check OOB (out of bag) error estimate and model performance on the training data set.
# OOB score is used instead of cross validation accuracy.
  
  # OOB error is near the top of the following output.
  # For random forests, OOB error makes cross validation unnecessary.
  prop.table(table(train$Species == rf$predicted))
  
  # Compute model performance on training data.
  
    # View training accuracy.
    print('Training Accuracy:')
    table(train$Species == train$pred)
    prop.table(table(train$Species == train$pred))
    
#####
# EXPORT TEST SET PREDICTIONS.

  # View test set accuracy.
  print('Test Set Accuracy:')
  table(test$Species == test$pred)
  prop.table(table(test$Species == test$pred))