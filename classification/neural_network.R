# This script creates and evaluates a neural network model in R for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering and parameter tuning.

# Load packages.
library(data.table)
library(nnet)
library(e1071)

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
  
  # Create NN.
  nn = nnet(model, 
            data = train, 
            size = 2)

  # Generate predictions for training and test data.
  train$pred = data.table(predict(nn,
                               newdata = train,
                               type = 'class'))
  test$pred = data.table(predict(nn, 
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
    
  # Create 'tune.control' object.
  tune_control = tune.control(cross = 2)
  
  # Execute cross validation.
  t = tune.nnet(model, 
                data = train, 
                size = 2, 
                tunecontrol = tune_control)
  
  # View estimated test set error computed by cross validation.
  print(t)

#####
# CHECK TEST SET ACCURACY.

  # View test set accuracy.
  print('Test Set Accuracy:')
  table(test$Species == test$pred)
  prop.table(table(test$Species == test$pred))
