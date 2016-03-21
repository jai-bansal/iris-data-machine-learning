# This script creates and evaluates an K Nearest Neighbor model in R for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering and parameter tuning.

# Load packages.
library(data.table)
library(kknn)

# Import data.
train = data.table(read.csv('train.csv',
                            header = T))
test = data.table(read.csv('test.csv', 
                           header = T))

#####
# SCALE FEATURES.
  
  # Scale variables in 'train'.
  Sepal.Length_train_scaled = scale(train$Sepal.Length)
  Sepal.Width_train_scaled = scale(train$Sepal.Width)
  Petal.Length_train_scaled = scale(train$Petal.Length)
  Petal.Width_train_scaled = scale(train$Petal.Width)
  
  # Scale dvariables in 'test' using mean and standard deviation derived from scaling variables in 'train'.
  Sepal.Length_test_scaled = (test$Sepal.Length - attr(Sepal.Length_train_scaled, 'scaled:center')) / attr(Sepal.Length_train_scaled, 'scaled:scale')
  Sepal.Width_test_scaled = (test$Sepal.Width - attr(Sepal.Width_train_scaled, 'scaled:center')) / attr(Sepal.Width_train_scaled, 'scaled:scale')
  Petal.Length_test_scaled = (test$Petal.Length - attr(Petal.Length_train_scaled, 'scaled:center')) / attr(Petal.Length_train_scaled, 'scaled:scale')
  Petal.Width_test_scaled = (test$Petal.Width - attr(Petal.Width_train_scaled, 'scaled:center')) / attr(Petal.Width_train_scaled, 'scaled:scale')
  
  # Create 'train_model' and 'test_model' which only include variables used in the model.
  train_model = data.table(Species = train$Species, 
                           Sepal.Length_scaled = Sepal.Length_train_scaled, 
                           Sepal.Width_scaled = Sepal.Width_train_scaled,
                           Petal.Length_scaled = Petal.Length_train_scaled,
                           Petal.Width_scaled = Petal.Width_train_scaled)

  test_model = data.table(Species = test$Species, 
                           Sepal.Length_scaled = Sepal.Length_test_scaled, 
                           Sepal.Width_scaled = Sepal.Width_test_scaled,
                           Petal.Length_scaled = Petal.Length_test_scaled,
                           Petal.Width_scaled = Petal.Width_test_scaled)
  
  # 'train_model' columns have incorrect names for some reason.
  setnames(train_model, 
           names(train_model), 
           c('Species', 'Sepal.Length_scaled', 'Sepal.Width_scaled', 'Petal.Length_scaled', 'Petal.Width_scaled'))
    
#####
# CREATE MODEL AND PREDICTIONS.
  
  # Set seed to ensure reproducibility.
  set.seed(1)
  
  # Define model.
  model = Species ~ Sepal.Length_scaled + Sepal.Width_scaled + Petal.Length_scaled + Petal.Width_scaled
  
  # Create model and generate predictions for training set.
  knn_train = kknn(formula = model, 
                      train = train_model, 
                      test = train_model)
  
  # Create model and generate predictions for test set.
  knn_test = kknn(formula = model, 
                  train = train_model,
                  test = test_model)

  # Generate predictions for training and test data.
  # Training and test predictions  were already computed when generating model.
  train$pred = data.table(knn_train$fitted.values)
  test$pred = data.table(knn_test$fitted.values)

#####
# CHECK TRAINING SET ACCURACY.

  # View training accuracy.
  print('Training Accuracy')
  print(table(train$Species == train$pred))
  print(prop.table(table(train$Species == train$pred)))

#####
# CROSS VALIDATION.
  
  # Conduct cross validation.
  cv = cv.kknn(model, 
               data = train_model, 
               kcv = 2)
  
  # View cross validation accuracy.
  cv = data.table(cv[[1]])
  print('Cross Validation Accuracy')
  print(table(cv$y == cv$yhat))
  print(prop.table(table(cv$y == cv$yhat)))

#####
# TEST SET ACCURACY.

  # View test set accuracy.
  print('Test Set Accuracy')
  print(table(test$Species == test$pred))
  print(prop.table(table(test$Species == test$pred)))