# This script creates and evaluates a neural network classification model in R for the 'iris' data.
# It uses the 'tensorflow' package.
# This iteration gets the model working as quickly as possible; there is no
# cross validation, feature engineering, or parameter tuning.

# LOAD LIBRARIES ----------------------------------------------------------
# This section loads libraries.
library(readr)
library(data.table)
library(dplyr)
library(tensorflow)

# IMPORT DATA -------------------------------------------------------------
# This section imports data.
# The data files are located in the 'R' branch file, NOT the 'classification' or 'regression' folders.
# The commands below assume the working directory is correctly set.
train = data.table(read_csv('train.csv'))
test = data.table(read_csv('test.csv'))

# PREP DATA ---------------------------------------------------------------
# This section preps the data for model training.

  # Get training and test data (not labels).
  train_data = as.matrix(select(train, 
                                -c(Species)))
  test_data = as.matrix(select(test, 
                               -c(Species)))
  
  # Get training and test labels.
  train_labels = as.matrix(train$Species)
  test_labels = as.matrix(test$Species)
  
# CREATE MODEL ------------------------------------------------------------
# This section creates a neural network classification model using 'tensorflow'.
  
  # Specify number of classes (output labels).
  classes = 3
  
  # Specify number of hidden nodes.
  hidden = 1000
  
  # Set # of iterations to run.
  steps = 10
  
  # Set random seed.
  tf$set_random_seed(20170907)
  
  # Input training and test data and labels.
  # I use constant data instead of placeholder data since the 'iris' data is really small.
  train_data = tf$constant(train_data, 
                           dtype = tf$float32)
  test_data = tf$constant(test_data, 
                          dtype = tf$float32)
  train_labels = tf$constant(train_labels)
  test_labels = tf$constant(test_labels)
  
  # Define 2 layers of weights and biases.
  w_1 = tf$Variable(tf$truncated_normal(shape(4, hidden)))
  b_1 = tf$Variable(tf$zeros(shape(hidden)))
  w_2 = tf$Variable(tf$truncated_normal(shape(hidden, classes)))
  b_2 = tf$Variable(tf$zeros(shape(classes)))
  
  # Specify training computation.
  mm_1 = tf$matmul(train_data, w_1) + b_1
  relu_1 = tf$nn$relu(mm_1)
  mm_2 = tf$matmul(relu_1, w_2) + b_2




  
  # Convert training and test set labels into one-hot vectors.
  model.matrix(~ Species - 1, 
               data = train)
  
  model.matrix(~ ZLSG3BW_TEXT - 1,
                                   data = agg_store)