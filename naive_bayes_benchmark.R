# This script creates and evaluates an adaboost model in R for the 'iris' data.
# This iteration is only to get the model up and running, so there is no feature engineering and parameter tuning.

# Load packages.
library(data.table)
library(e1071)

# Import data.
train = data.table(read.csv('../input/train.csv',
                            header = T))
test = data.table(read.csv('../input/test.csv', 
                           header = T))

#####
# PRE-PROCESSING AND FEATURE ENGINEERING.
  
  # Rename columns of testing and training data set.
  setnames(train, names(train), c('date', 'category_predict', 'description_ignore', 'day_of_week', 'pd_district', 'resolution', 'address', 'x', 'y'))
  setnames(test, names(test), c('id', 'date', 'day_of_week', 'pd_district', 'address', 'x', 'y'))
  
  # Get hour of each crime.
  train$hour = as.numeric(substr(train$date, 12, 13))
  test$hour = as.numeric(substr(test$date, 12, 13))
  
  # Create data subset to train model on (training on full data set is too time-intensive).
  # 'subset' contains at least 1 example of every crime category.
  subset = train[c(1:30000, 30024, 33954, 37298, 41097, 41980, 44479, 48707, 48837, 49306, 53715, 93717, 102637, 102645, 102678, 102919, 103517, 103712, 107734, 148476, 148476, 
                   192191, 205046, 252094, 279792, 316491, 317527, 332821, 337881)]
  
#####
# CREATE MODEL AND PREDICTIONS.
  
  # Set seed to ensure reproducibility.
  set.seed(1)
  
  # Define model.
  model = category_predict ~ day_of_week + pd_district + x + y + hour
  
  # Create model and generate predictions for training set.
  nb = naiveBayes(model, 
                  data = subset)
  
  # Generate predictions for training and test data.
  # For the training data, I only want to compute accuracy.
  # For the test data, I need to put predictions in a specific format for submission, as specified by Kaggle.com.
  subset$pred = data.table(predict(nb, 
                                   newdata = subset, 
                                   type = 'class'))
  test_pred = data.table(predict(nb, 
                                 newdata = test, 
                                 type = 'raw'))

#####
# CHECK TRAINING SET ACCURACY.
  
  # View training accuracy.
  print('Training Accuracy:')
  table(subset$category_predict == subset$pred)
  prop.table(table(subset$category_predict == subset$pred))

#####
# CROSS VALIDATION.
    
  # Create 'tune.control' object.
  # Certain crime categories don't have many occurences in 'subset' so I keep the number of cross validation partitions low.
  tune_control = tune.control(cross = 2)
  
  # Execute cross validation.
  t = tune(method = 'naiveBayes', 
           model, 
           data = subset, 
           size = 2, 
           tunecontrol = tune_control)
  
  # View estimated test set error computed by cross validation.
  print('Cross Validation Error: ')
  print(t)

#####
# EXPORT TEST SET PREDICTIONS.

  # Predictions must be formatted as specified on Kaggle.com.
  # This is done for test data only.
  
    # Add 'test$id' to 'test_pred'.
    test_pred$id = test$id
  
  # Create csv file of test predictions.
  # This is commented out for now, since I don't actually want to create a csv.
  # write.csv(test_pred, 'test_pred_naive_bayes_benchmark.csv', row.names = F)